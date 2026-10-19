"""Nested, stratified evaluation with training-only threshold selection."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from .data import FEATURES, TARGET, load_data
from .metrics import bootstrap_intervals, choose_threshold, metrics
from .models import DEPLOYED, candidates
from .plots import evaluation_plots


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def tune_and_fit(estimator, grid, X, y, seed, jobs, sensitivity):
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator, grid, scoring="neg_brier_score", cv=inner, n_jobs=jobs, refit=True, error_score="raise"
    )
    search.fit(X, y)
    # Hyperparameter selection reuses development labels. This threshold estimate is
    # NOT reported as unbiased; only outer-fold predictions support reported metrics.
    oof = cross_val_predict(search.best_estimator_, X, y, cv=inner, method="predict_proba", n_jobs=jobs)[:, 1]
    threshold = choose_threshold(y, oof, sensitivity)
    return search.best_estimator_, threshold, search.best_params_


def stress_scenarios(X, seed):
    cases = {"original": X.copy()}
    for column in ["Cholesterol", "RestingBP", "ST_Slope"]:
        masked = X.copy()
        masked[column] = np.nan
        cases[f"missing_{column}"] = masked
    for fraction in [0.10, 0.30]:
        rng = np.random.default_rng(seed)
        mask = rng.random(X.shape) < fraction
        cases[f"random_missing_{int(fraction * 100)}pct"] = X.mask(mask)
    return cases


def subgroup_scores(X, predictions):
    rows = []
    groupings = {
        "Sex": X.Sex.fillna("Missing"),
        "Age": pd.cut(X.Age, [0, 45, 60, 150], labels=["18–45", "46–60", "61+"]),
    }
    for name, groups in groupings.items():
        for group in groups.dropna().unique():
            subset = predictions.loc[groups == group]
            if len(subset):
                scores = metrics(subset.y, subset.probability, subset.threshold)
                rows.append(
                    {"attribute": name, "group": str(group), "small_group": len(subset) < 30, **scores}
                )
    return pd.DataFrame(rows)


def run_experiment(
    data_path, output, folds=5, seed=42, jobs=1, bootstrap=1000, smoke=False, sensitivity=0.90
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("This run already exists. Choose a new output folder to preserve provenance.")
    df, audit = load_data(data_path)
    X, y = df[FEATURES], df[TARGET]
    if y.value_counts().min() < max(30, folds * 3):
        raise ValueError("Too few examples per class for the nested protocol (minimum 30).")
    json_write(output / "data_audit.json", audit)
    catalog = candidates(seed, trees=30 if smoke else 150)
    if smoke:
        catalog = {k: v for k, v in catalog.items() if k in ["logistic", DEPLOYED]}
        folds = 2
    outer = StratifiedKFold(folds, shuffle=True, random_state=seed)
    prediction_rows, fold_rows, stress_rows = [], [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        for name, (estimator, grid) in catalog.items():
            print(f"Outer fold {fold}/{folds}: {name}", flush=True)
            model, threshold, parameters = tune_and_fit(
                estimator, grid, X_train, y_train, seed + fold, jobs, sensitivity
            )
            proba = model.predict_proba(X_test)[:, 1]
            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "threshold": threshold,
                    "parameters": json.dumps(parameters),
                    **metrics(y_test, proba, threshold),
                }
            )
            prediction_rows.extend(
                {
                    "row_id": int(i),
                    "fold": fold,
                    "model": name,
                    "y": int(label),
                    "probability": float(p),
                    "threshold": threshold,
                }
                for i, label, p in zip(test_idx, y_test, proba)
            )
            if name == DEPLOYED:
                for scenario, masked in stress_scenarios(X_test, seed + fold).items():
                    p = model.predict_proba(masked)[:, 1]
                    stress_rows.extend(
                        {
                            "row_id": int(i),
                            "scenario": scenario,
                            "y": int(label),
                            "probability": float(prob),
                            "threshold": threshold,
                        }
                        for i, label, prob in zip(test_idx, y_test, p)
                    )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    summaries, intervals = [], {}
    for name, rows in predictions.groupby("model"):
        summaries.append({"model": name, **metrics(rows.y, rows.probability, rows.threshold)})
        intervals[name] = bootstrap_intervals(
            rows.y, rows.probability, rows.threshold, repeats=100 if smoke else bootstrap, seed=seed
        )
    pd.DataFrame(summaries).to_csv(output / "model_comparison.csv", index=False)
    json_write(output / "confidence_intervals.json", intervals)
    stress = pd.DataFrame(stress_rows)
    pd.DataFrame(
        [
            {"scenario": name, **metrics(rows.y, rows.probability, rows.threshold)}
            for name, rows in stress.groupby("scenario")
        ]
    ).to_csv(output / "robustness.csv", index=False)
    deployed_rows = predictions[predictions.model == DEPLOYED].set_index("row_id").sort_index()
    subgroup_scores(X, deployed_rows).to_csv(output / "subgroups.csv", index=False)
    evaluation_plots(predictions, output)

    print(
        "Fitting the pre-specified calibrated voting model on all available development data...", flush=True
    )
    estimator, grid = catalog[DEPLOYED]
    final, threshold, parameters = tune_and_fit(estimator, grid, X, y, seed, jobs, sensitivity)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "shap", "streamlit", "joblib"]
    }
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "outer_folds": folds,
        "inner_folds": 3,
        "calibration_folds": 3,
        "smoke_run": smoke,
        "rows": len(df),
        "dataset_sha256": audit["sha256"],
        "deployed_model": DEPLOYED,
        "threshold": threshold,
        "threshold_minimum_development_sensitivity": sensitivity,
        "parameters": parameters,
        "selection_scoring": "neg_brier_score",
        "versions": versions,
        "python": platform.python_version(),
        "protocol": "Nested stratified CV; calibration, hyperparameters and thresholds inside outer training folds.",
        "limitations": [
            "Internal validation on a public retrospective dataset, not external or clinical validation.",
            "The original minor project has already used this dataset; this is not a newly collected holdout.",
            "No patient/site identifiers: distinct records cannot be proven to be distinct patients/sites.",
            "Bootstrap intervals condition on OOF predictions and do not capture full training variability.",
            "Minimum development sensitivity is an experimental target, not a guaranteed deployment property.",
            "Masking experiments simulate missingness; they do not establish real-world missingness performance.",
            "Final all-data fit is for demonstration; its training performance is not reported as validation.",
        ],
    }
    fingerprints = pd.util.hash_pandas_object(X, index=False).astype(str).tolist()
    bundle = {
        "model": final,
        "threshold": threshold,
        "manifest": manifest,
        "background": X.sample(min(30, len(X)), random_state=seed),
        "training_fingerprints": fingerprints,
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X.select_dtypes(include="number")},
    }
    joblib.dump(bundle, output / "model.joblib", compress=3)
    manifest["model_sha256"] = hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
    json_write(output / "manifest.json", manifest)
    from .reporting import experiment_report

    experiment_report(output)
    print(f"Completed: {output.resolve()}", flush=True)
    return manifest


def external_evaluate(model_path, data_path, output):
    """Frozen-model evaluation: refuses exact overlap; cannot prove patient independence."""
    bundle = joblib.load(model_path)
    df, audit = load_data(data_path)
    X = df[FEATURES]
    fingerprints = set(pd.util.hash_pandas_object(X, index=False).astype(str))
    overlap = fingerprints & set(bundle["training_fingerprints"])
    if overlap:
        raise ValueError(
            f"External evaluation refused: {len(overlap)} exact feature records overlap training."
        )
    p = bundle["model"].predict_proba(X)[:, 1]
    result = {
        "audit": audit,
        "metrics": metrics(df[TARGET], p, bundle["threshold"]),
        "confidence_intervals": bootstrap_intervals(df[TARGET], p, bundle["threshold"]),
        "note": "Exact-overlap check passed. Confirm patient/site independence and label equivalence separately.",
    }
    json_write(output, result)
    return result

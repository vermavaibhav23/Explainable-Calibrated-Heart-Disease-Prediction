"""Survey-weighted development and temporal evaluation of cardiovascular mortality risk."""

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold

from .metrics import metrics
from .nhanes import ENDPOINT, SETS
from .risk import RiskModel, calibrated, catalog, logit, save_json


def weighted_metrics(y, p, w, threshold):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
    positive = p >= threshold
    tp, fp = w[(y == 1) & positive].sum(), w[(y == 0) & positive].sum()
    fn, tn = w[(y == 1) & ~positive].sum(), w[(y == 0) & ~positive].sum()
    return {"roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
            "pr_auc_ap": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "accuracy": float((tp + tn) / w.sum()), "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)), "precision": float(tp / (tp + fp)) if tp + fp else None,
            "prevalence": float(np.average(y, weights=w))}


def fit_calibrator(method, p, y, w):
    if method == "none":
        return None
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, y, sample_weight=w)
    return LogisticRegression(C=10, max_iter=2000).fit(logit(p), y, sample_weight=w)


def weighted_threshold(y, p, w, target=0.8):
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
    p, w = p[y == 1], w[y == 1]
    order = np.argsort(p)
    index = np.searchsorted(np.cumsum(w[order]), (1 - target) * w.sum(), side="right")
    return float(p[order[min(index, len(p)-1)]])


def fit_weighted(model, X, y, w):
    return model.fit(X, y, **{model.steps[-1][0] + "__sample_weight": np.asarray(w)})


def weighted_oof(model, X, y, w, splits):
    probability = np.zeros(len(X))
    for a, b in splits:
        fitted = fit_weighted(clone(model), X.iloc[a], y.iloc[a], w.iloc[a])
        probability[b] = fitted.predict_proba(X.iloc[b])[:, 1]
    return probability


def design_bootstrap(data, p, threshold, repeats=1000):
    """Rao-Wu-style n_h-1 PSU resampling for this design's two-PSU strata.

    The common weight multiplier of two cancels in all normalized metrics.
    Intervals condition on the model and omit finite-population corrections.
    """
    data = data.reset_index(drop=True)
    groups = {s: {psu: part.index.to_numpy() for psu, part in group.groupby("psu")}
              for s, group in data.groupby("stratum")}
    if any(len(group) != 2 for group in groups.values()):
        raise ValueError("This rescaled bootstrap requires exactly two PSUs per stratum.")
    rng = np.random.default_rng(2026)
    results = []
    for _ in range(repeats):
        idx = np.concatenate([group[k] for group in groups.values()
                              for k in rng.choice(list(group), len(group) - 1, replace=True)])
        if data.target.iloc[idx].nunique() == 2:
            results.append(weighted_metrics(data.target.iloc[idx], p[idx], data.weight.iloc[idx], threshold))
    return {k: np.quantile([r[k] for r in results], [0.025, 0.975]).tolist()
            for k in ["roc_auc", "brier", "sensitivity", "specificity", "accuracy", "pr_auc_ap"]}


def run(data_path, output, iterations=6, jobs=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(data_path)
    development, test = data.loc[data.phase.eq(1)], data.loc[data.phase.eq(2)]
    if set(development.family) & set(test.family):
        raise ValueError("Family overlap across temporal partitions.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    a, b = next(splitter.split(development, groups=development.family))
    train, calibration = development.iloc[a], development.iloc[b]
    assert not set(train.family) & set(calibration.family)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=2026)
    splits = list(cv.split(train, train.target, train.family))
    cal_splits = list(cv.split(calibration, calibration.target, calibration.family))
    protocol = {"endpoint": ENDPOINT, "development_phase": "1988–1991", "test_phase": "1991–1994",
                "seed": 2026, "cv": "5-fold family-grouped, stratified", "iterations": iterations,
                "features": SETS, "training_weight": "phase-specific MEC weight, normalized to mean 1",
                "selection": "weighted development OOF AUROC; top-three blend competes with best single",
                "calibration": "weighted Brier over five family-grouped calibration folds",
                "threshold": "weighted calibration OOF sensitivity target 0.80",
                "sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                "partitions": {"train_ids": train.id.tolist(), "calibration_ids": calibration.id.tolist(),
                               "test_ids": test.id.tolist()},
                "counts": {"train": len(train), "calibration": len(calibration), "test": len(test)},
                "limitations": ["Historical US cohort; no contemporary independent validation",
                                "Public death cause/time sometimes perturbed for confidentiality",
                                "Fatal heart/cerebrovascular events only; not first MI or all CVD",
                                "Includes people with baseline disease; ages 40–89"]}
    save_json(output / "protocol.json", protocol)
    summary = {}
    for feature_set, features in SETS.items():
        folder = output / feature_set
        folder.mkdir()
        X, y = train[features], train.target
        w = train.weight / train.weight.mean()
        fitted, oof, rows = {}, {}, []

        def scorer(estimator, inputs, labels):
            return roc_auc_score(labels, estimator.predict_proba(inputs)[:, 1], sample_weight=w.loc[inputs.index])

        for name, (estimator, space) in catalog(2026).items():
            print(f"NHANES {feature_set}: {name}", flush=True)
            search = RandomizedSearchCV(estimator, space, n_iter=min(iterations, int(np.prod([len(v) for v in space.values()]))),
                                        cv=splits, scoring=scorer, random_state=2026, n_jobs=jobs, error_score="raise")
            search.fit(X, y, **{estimator.steps[-1][0] + "__sample_weight": w.to_numpy()})
            pd.DataFrame(search.cv_results_).to_csv(folder / f"search_{name}.csv", index=False)
            fitted[name] = search.best_estimator_
            oof[name] = weighted_oof(search.best_estimator_, X, y, w, splits)
            auc = roc_auc_score(y, oof[name], sample_weight=w)
            rows.append({"model": name, "development_auc": auc, "parameters": search.best_params_})
            print(f"  weighted development AUROC: {auc:.4f}", flush=True)
        rows.sort(key=lambda row: row["development_auc"], reverse=True)
        top = [r["model"] for r in rows[:3]]
        blended = np.mean([oof[n] for n in top], axis=0)
        auc = roc_auc_score(y, blended, sample_weight=w)
        selected = top if auc > rows[0]["development_auc"] else [top[0]]
        rows.append({"model": "top3_blend", "development_auc": auc, "parameters": top})
        save_json(folder / "development_search.json", rows)
        raw = np.mean([fitted[n].predict_proba(calibration[features])[:, 1] for n in selected], axis=0)
        yc, wc = calibration.target, calibration.weight / calibration.weight.mean()
        cal_results, cal_oof = [], {}
        for method in ["none", "sigmoid", "isotonic"]:
            p = np.zeros(len(calibration))
            for a, b in cal_splits:
                c = fit_calibrator(method, raw[a], yc.iloc[a], wc.iloc[a])
                p[b] = calibrated(method, c, raw[b])
            cal_oof[method] = p
            cal_results.append({"method": method, "brier": brier_score_loss(yc, p, sample_weight=wc)})
        method = min(cal_results, key=lambda row: row["brier"])["method"]
        threshold = weighted_threshold(yc, cal_oof[method], wc)
        model = RiskModel([fitted[n] for n in selected], features, method,
                          fit_calibrator(method, raw, yc, wc), threshold)
        joblib.dump(model, folder / "model.joblib")
        save_json(folder / "selection.json", {"models": selected, "calibration": method,
                                              "calibration_cv": cal_results, "threshold": threshold})
        train[features].sample(30, random_state=2026).to_csv(folder / "explanation_background.csv", index=False)
        # The later phase is evaluated only after selection, calibration and threshold are frozen.
        p = model.predict_proba(test[features])[:, 1]
        result = {"models": selected, "calibration": method, "threshold": threshold,
                  "weighted_screening": weighted_metrics(test.target, p, test.weight, threshold),
                  "weighted_at_0_5": weighted_metrics(test.target, p, test.weight, 0.5),
                  "unweighted_screening": metrics(test.target, p, threshold),
                  "unweighted_at_0_5": metrics(test.target, p, 0.5),
                  "ci95_design_bootstrap": design_bootstrap(test, p, threshold),
                  "weighted_constant_baseline": weighted_metrics(test.target, np.full(len(test), np.average(y, weights=w)), test.weight, 0.5)}
        save_json(folder / "test_metrics.json", result)
        predictions = test[["id", "target", "weight", "stratum", "psu", "male", "age"]].copy()
        predictions["probability"] = p
        predictions.to_csv(folder / "test_predictions.csv", index=False)
        subgroup_rows = []
        for group, subset in predictions.groupby("male"):
            subgroup_rows.append({"male": int(group), "n": len(subset), "events": int(subset.target.sum()),
                                  **weighted_metrics(subset.target, subset.probability, subset.weight, threshold)})
        save_json(folder / "subgroups.json", subgroup_rows)
        summary[feature_set] = result
        print(f"{feature_set} temporal test: {result['weighted_screening']}", flush=True)
    save_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/nhanes3_risk_standardized.csv")
    parser.add_argument("--output", default="artifacts/ecg-risk-v3")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    run(args.data, args.output, args.iterations, args.jobs)


if __name__ == "__main__":
    from heart_research.ecg_risk import main as entry
    entry()

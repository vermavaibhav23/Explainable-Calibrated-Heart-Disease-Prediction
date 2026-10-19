"""Affordable 10-year CHD research benchmark. Never an MI-only clinical model."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .metrics import bootstrap_intervals, metrics

TARGET = "TenYearCHD"
NO_LAB = ["male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke",
          "prevalentHyp", "diabetes", "sysBP", "diaBP", "BMI", "heartRate"]
FEATURE_SETS = {"no_lab": NO_LAB, "basic_lab": NO_LAB + ["totChol", "glucose"]}
SOURCE = "https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/resources/framingham/"


class CatBoostCompat(ClassifierMixin, BaseEstimator):
    """Explicit sklearn estimator contract for CatBoost 1.2.8 with sklearn 1.8."""

    def __init__(self, iterations=250, depth=4, learning_rate=0.05, l2_leaf_reg=3, random_seed=2026):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier
        self.model_ = CatBoostClassifier(**self.get_params(), verbose=False,
                                        allow_writing_files=False, thread_count=1)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X).ravel()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_risk_data(path):
    data = pd.read_csv(path)
    required = FEATURE_SETS["basic_lab"] + [TARGET]
    if not set(required).issubset(data.columns):
        raise ValueError("Expected MIT Framingham schema with TenYearCHD outcome.")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Outcome must be observed and binary.")
    data = data[required].apply(pd.to_numeric, errors="raise")
    # Shared baseline profiles must never straddle development and test sets.
    keys = pd.util.hash_pandas_object(data.drop(columns=TARGET), index=False)
    if data.groupby(keys)[TARGET].nunique().max() > 1:
        raise ValueError("Conflicting outcomes for identical baseline profiles.")
    return data.loc[~keys.duplicated()].reset_index(drop=True)


def catalog(seed):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    def pipe(model, spline=False):
        steps = [SimpleImputer(strategy="median", add_indicator=True)]
        if spline:
            steps.append(SplineTransformer(n_knots=4, degree=3, extrapolation="linear"))
        return make_pipeline(*steps, StandardScaler(), model)

    lr = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]}
    forest = {"n_estimators": 350, "n_jobs": 1, "random_state": seed}
    return {
        "logistic": (pipe(LogisticRegression(max_iter=3000)), lr),
        "spline_logistic": (pipe(LogisticRegression(max_iter=3000), True), lr),
        "random_forest": (pipe(RandomForestClassifier(**forest)), {
            "randomforestclassifier__min_samples_leaf": [5, 15, 30, 60],
            "randomforestclassifier__max_features": [0.5, 1.0],
            "randomforestclassifier__max_depth": [4, 8, None]}),
        "extra_trees": (pipe(ExtraTreesClassifier(**forest)), {
            "extratreesclassifier__min_samples_leaf": [5, 15, 30, 60],
            "extratreesclassifier__max_features": [0.5, 1.0],
            "extratreesclassifier__max_depth": [4, 8, None]}),
        "hist_gradient": (pipe(HistGradientBoostingClassifier(random_state=seed)), {
            "histgradientboostingclassifier__max_leaf_nodes": [5, 10, 20],
            "histgradientboostingclassifier__l2_regularization": [1, 10, 50],
            "histgradientboostingclassifier__learning_rate": [0.02, 0.05, 0.1]}),
        "catboost": (pipe(CatBoostCompat(random_seed=seed)), {
            "catboostcompat__iterations": [250, 500],
            "catboostcompat__depth": [3, 4, 6],
            "catboostcompat__learning_rate": [0.02, 0.05],
            "catboostcompat__l2_leaf_reg": [3, 10, 30]}),
        "xgboost": (pipe(XGBClassifier(n_jobs=1, random_state=seed, eval_metric="logloss")), {
            "xgbclassifier__n_estimators": [200, 400], "xgbclassifier__max_depth": [2, 3, 5],
            "xgbclassifier__learning_rate": [0.02, 0.05],
            "xgbclassifier__min_child_weight": [5, 20], "xgbclassifier__reg_lambda": [5, 20]}),
        "lightgbm": (pipe(LGBMClassifier(n_jobs=1, random_state=seed, verbosity=-1)), {
            "lgbmclassifier__n_estimators": [200, 400], "lgbmclassifier__num_leaves": [5, 10, 20],
            "lgbmclassifier__learning_rate": [0.02, 0.05],
            "lgbmclassifier__min_child_samples": [30, 70], "lgbmclassifier__reg_lambda": [5, 20]}),
    }


def calibrator_fit(method, probability, y):
    if method == "none":
        return None
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(probability, y)
    return LogisticRegression(C=10, max_iter=1000).fit(logit(probability), y)


def logit(probability):
    p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrated(method, model, probability):
    if method == "none":
        return probability
    if method == "isotonic":
        return model.predict(probability)
    return model.predict_proba(logit(probability))[:, 1]


class RiskModel(ClassifierMixin, BaseEstimator):
    """Persisted inference contract: ensemble, schema, calibration and threshold."""

    def __init__(self, models, features, method="none", calibration=None, threshold=0.5):
        self.models = models
        self.features = features
        self.method = method
        self.calibration = calibration
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        X = X[self.features]
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        p = calibrated(self.method, self.calibration, p)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def threshold_for(y, p, sensitivity=0.8):
    # Largest attainable threshold meeting the development sensitivity target.
    positive = np.sort(np.asarray(p)[np.asarray(y) == 1])
    return float(positive[int(np.floor((1 - sensitivity) * len(positive)))])


def run(data_path, output, seed=2026, iterations=10, jobs=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_risk_data(data_path)
    y = data[TARGET]
    development, test = train_test_split(np.arange(len(data)), test_size=0.2,
                                         stratify=y, random_state=seed)
    train, calibration = train_test_split(development, test_size=0.25,
                                          stratify=y.iloc[development], random_state=seed + 1)
    # Persist protocol before any training or test evaluation. Never choose a seed by its results.
    protocol = {"endpoint": "10-year coronary heart disease (not MI-only)", "seed": seed,
                "source": SOURCE, "sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                "rows": len(data), "splits": {"train": train.tolist(), "calibration": calibration.tolist(),
                                               "test": test.tolist()},
                "search_iterations_per_family": iterations, "cv_folds": 5,
                "selection": "development CV AUROC; top-three uniform blend competes using development OOF AUROC",
                "calibration_selection": "five-fold calibration-partition Brier score",
                "threshold": "calibration OOF sensitivity target 0.80; not a treatment threshold",
                "limitations": ["teaching-source provenance; research only", "no external validation",
                                "unknown censoring and participant identifiers", "historical US cohort"],
                "versions": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scikit-learn", "catboost", "xgboost", "lightgbm"]}}
    save_json(output / "protocol.json", protocol)
    summary = {}
    for feature_set, features in FEATURE_SETS.items():
        folder = output / feature_set
        folder.mkdir()
        X = data[features]
        Xt, yt = X.iloc[train], y.iloc[train]
        Xc, yc = X.iloc[calibration], y.iloc[calibration]
        cv = StratifiedKFold(5, shuffle=True, random_state=seed + 2)
        fitted, oof, rows = {}, {}, []
        for name, (estimator, space) in catalog(seed).items():
            print(f"{feature_set}: searching {name}", flush=True)
            count = int(np.prod([len(v) for v in space.values()]))
            search = RandomizedSearchCV(estimator, space, n_iter=min(iterations, count), cv=cv,
                                        scoring="roc_auc", n_jobs=jobs, random_state=seed,
                                        error_score="raise", return_train_score=False)
            search.fit(Xt, yt)
            pd.DataFrame(search.cv_results_).to_csv(folder / f"search_{name}.csv", index=False)
            fitted[name] = search.best_estimator_
            oof[name] = cross_val_predict(search.best_estimator_, Xt, yt, cv=cv,
                                          method="predict_proba", n_jobs=jobs)[:, 1]
            rows.append({"model": name, "development_auc": roc_auc_score(yt, oof[name]),
                         "cv_auc_sd": search.cv_results_["std_test_score"][search.best_index_],
                         "parameters": search.best_params_})
            print(f"  development AUROC {rows[-1]['development_auc']:.4f}", flush=True)
        rows.sort(key=lambda r: r["development_auc"], reverse=True)
        top = [r["model"] for r in rows[:3]]
        blend = np.mean([oof[n] for n in top], axis=0)
        blend_auc = roc_auc_score(yt, blend)
        selected = top if blend_auc > rows[0]["development_auc"] else [top[0]]
        rows.append({"model": "top3_blend", "development_auc": blend_auc, "parameters": top})
        save_json(folder / "development_search.json", rows)
        # Development scores are selection-biased; only the untouched test supports performance claims.
        raw = np.mean([fitted[n].predict_proba(Xc)[:, 1] for n in selected], axis=0)
        calibration_rows, cp = [], {}
        for method in ["none", "sigmoid", "isotonic"]:
            p = np.zeros(len(yc))
            for a, b in cv.split(raw, yc):
                model = calibrator_fit(method, raw[a], yc.iloc[a])
                p[b] = calibrated(method, model, raw[b])
            cp[method] = p
            calibration_rows.append({"method": method, "brier": brier_score_loss(yc, p)})
        method = min(calibration_rows, key=lambda r: r["brier"])["method"]
        threshold = threshold_for(yc, cp[method])
        model = RiskModel([fitted[n] for n in selected], features, method,
                          calibrator_fit(method, raw, yc), threshold)
        # Freeze model/threshold before examining holdout predictions.
        joblib.dump(model, folder / "model.joblib")
        save_json(folder / "selection.json", {"models": selected, "calibration": method,
                                              "calibration_cv": calibration_rows, "threshold": threshold})
        Xtest, ytest = X.iloc[test], y.iloc[test]
        probability = model.predict_proba(Xtest)[:, 1]
        result = {"selected_models": selected, "calibration": method, "threshold": threshold,
                  "screening": metrics(ytest, probability, threshold),
                  "threshold_0_5": metrics(ytest, probability, 0.5),
                  "constant_prevalence_baseline": metrics(ytest, np.full(len(test), yt.mean()), 0.5),
                  "ci95_screening": bootstrap_intervals(ytest, probability, threshold, 1000, seed)}
        save_json(folder / "holdout_metrics.json", result)
        pd.DataFrame({"row": test, "y": ytest.to_numpy(), "probability": probability}).to_csv(
            folder / "holdout_predictions.csv", index=False)
        groups = []
        for column, labels in {"male": Xtest.male.map({0: "female", 1: "male"}),
                               "age": pd.cut(Xtest.age, [0, 45, 60, 120])}.items():
            for label in labels.dropna().unique():
                mask = np.asarray(labels == label)
                groups.append({"attribute": column, "group": str(label),
                               **metrics(ytest[mask], probability[mask], threshold)})
        save_json(folder / "subgroups.json", groups)
        rng = np.random.default_rng(seed)
        stress = []
        for fraction in [0.1, 0.3]:
            masked = Xtest.mask(rng.random(Xtest.shape) < fraction)
            stress.append({"random_missing_fraction": fraction,
                           **metrics(ytest, model.predict_proba(masked)[:, 1], threshold)})
        save_json(folder / "missingness_stress.json", stress)
        curves = []
        for cutoff in np.linspace(0.05, 0.30, 26):
            scores = metrics(ytest, probability, cutoff)
            curves.append({"threshold": cutoff, "model_net_benefit":
                           (scores["tp"] - scores["fp"] * cutoff / (1 - cutoff)) / len(test),
                           "all_net_benefit": float(ytest.mean() - (1-ytest.mean()) * cutoff/(1-cutoff)),
                           "none_net_benefit": 0.0})
        pd.DataFrame(curves).to_csv(folder / "decision_curve.csv", index=False)
        summary[feature_set] = result
        print(f"{feature_set} holdout: {json.dumps(result['screening'])}", flush=True)
    save_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    run(args.data, args.output, iterations=args.iterations, jobs=args.jobs)


if __name__ == "__main__":
    # Persist classes under their importable module name, not __main__.
    from heart_research.risk import main as entry
    entry()

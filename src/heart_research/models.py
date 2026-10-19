"""Every learned transform is fitted inside the estimator's CV boundary."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .data import CATEGORIES, FEATURES, NUMERIC

INTERACTIONS = ["age_chol", "age_bp", "chol_fbs", "oldpeak_hr"]
REDUCED = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS"]
DEPLOYED = "calibrated_vote"


class ClinicalFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, interactions=True, reduced=False):
        self.interactions = interactions
        self.reduced = reduced

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        result = X[FEATURES].copy()
        for name in ["RestingBP", "Cholesterol"]:
            result[name] = result[name].replace(0, np.nan)
        if self.reduced:
            return result[REDUCED]
        if self.interactions:
            result["age_chol"] = result.Age * result.Cholesterol
            result["age_bp"] = result.Age * result.RestingBP
            result["chol_fbs"] = result.Cholesterol * result.FastingBS
            result["oldpeak_hr"] = result.Oldpeak * result.MaxHR
        return result


def pipeline(estimator, interactions=True, reduced=False):
    numeric = [c for c in NUMERIC if not reduced or c in REDUCED]
    if interactions and not reduced:
        numeric += INTERACTIONS
    categorical = [c for c in CATEGORIES if not reduced or c in REDUCED]
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [("features", ClinicalFeatures(interactions, reduced)), ("prep", prep), ("model", estimator)]
    )


def candidates(seed=42, trees=150):
    lr = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
    et = ExtraTreesClassifier(
        n_estimators=trees, max_depth=5, min_samples_leaf=4, max_features="sqrt", n_jobs=1, random_state=seed
    )
    svm = SVC(C=1, probability=True, random_state=seed)
    vote = VotingClassifier([("lr", lr), ("et", et), ("svm", svm)], voting="soft", n_jobs=1)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)

    def calibrated(interactions=True, reduced=False):
        return CalibratedClassifierCV(
            pipeline(vote, interactions, reduced), method="sigmoid", cv=cv, ensemble=False, n_jobs=1
        )

    # Grids fixed before looking at validation performance.
    return {
        "logistic": (pipeline(lr), {"model__C": [0.1, 1.0]}),
        "extra_trees": (pipeline(et), {"model__max_depth": [5, None]}),
        "svm": (pipeline(svm), {"model__C": [0.5, 2.0]}),
        "soft_vote": (pipeline(vote), {"model__weights": [None, [1, 2, 1]]}),
        DEPLOYED: (calibrated(), {"estimator__model__weights": [None, [1, 2, 1]]}),
        "calibrated_no_interactions": (calibrated(False), {"estimator__model__weights": [None]}),
        "calibrated_reduced_inputs": (calibrated(False, True), {"estimator__model__weights": [None]}),
    }

import numpy as np
import pandas as pd
import pytest

from heart_research.risk import calibrated, calibrator_fit, load_risk_data, threshold_for
from heart_research.risk_service import validate_record


@pytest.fixture
def record():
    return dict(male=0, age=50, currentSmoker=0, cigsPerDay=0, BPMeds=0,
                prevalentStroke=0, prevalentHyp=0, diabetes=0, sysBP=120, diaBP=80, BMI=24, heartRate=72)


@pytest.mark.parametrize("change", [{"age": 20}, {"sysBP": 60}, {"male": 2}, {"BMI": float("nan")},
                                    {"diaBP": 130}, {"cigsPerDay": 5}, {"sysBP": None}, {"age": "50"}])
def test_refuse_invalid_measurements(record, change):
    with pytest.raises(ValueError):
        validate_record(record | change, "no_lab")


def test_no_lab_does_not_accept_laboratory_or_outcome(record):
    for field in ["glucose", "totChol", "TenYearCHD"]:
        with pytest.raises(ValueError):
            validate_record(record | {field: 1}, "no_lab")


def test_optional_missingness_is_explicit(record):
    X, missing = validate_record(record | {"BPMeds": None}, "no_lab")
    assert missing == ["BPMeds"] and X.BPMeds.isna().all()


def test_conflicting_duplicates_rejected(tmp_path, record):
    rows = [record | {"totChol": 200, "glucose": 90, "TenYearCHD": y} for y in [0, 1]]
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        load_risk_data(path)


def test_threshold_keeps_target_sensitivity():
    y = np.array([0] * 10 + [1] * 10)
    p = np.linspace(0, 1, 20)
    threshold = threshold_for(y, p, 0.8)
    assert np.mean(p[y == 1] >= threshold) >= 0.8


@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_calibration_valid_probabilities(method):
    p = np.linspace(0.01, 0.99, 30)
    y = np.array([0, 0, 1] * 10)
    model = calibrator_fit(method, p, y)
    result = calibrated(method, model, np.array([0, 0.2, 1]))
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()


def test_catboost_adapter_supports_sklearn_scoring():
    pytest.importorskip("catboost")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    from heart_research.risk import CatBoostCompat
    X, y = make_classification(n_samples=50, n_features=5, random_state=8)
    scores = cross_val_score(CatBoostCompat(iterations=2), X, y, cv=2, scoring="roc_auc")
    assert np.isfinite(scores).all()

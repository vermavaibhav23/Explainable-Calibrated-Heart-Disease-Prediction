"""Regression tests for leakage boundaries, schema, and explanation fidelity."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from heart_research.data import FEATURES, load_data, validate_frame
from heart_research.explain import decode, encode, explain_records
from heart_research.metrics import choose_threshold, metrics
from heart_research.models import ClinicalFeatures, candidates, pipeline


@pytest.fixture
def records():
    rng = np.random.default_rng(7)
    n = 90
    return pd.DataFrame(
        {
            "Age": rng.integers(30, 75, n),
            "Sex": rng.choice(["M", "F"], n),
            "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], n),
            "RestingBP": rng.integers(100, 180, n),
            "Cholesterol": rng.integers(120, 330, n),
            "FastingBS": rng.integers(0, 2, n),
            "RestingECG": rng.choice(["Normal", "ST"], n),
            "MaxHR": rng.integers(90, 190, n),
            "ExerciseAngina": rng.choice(["Y", "N"], n),
            "Oldpeak": rng.uniform(0, 4, n),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], n),
        }
    )[FEATURES]


def test_engineered_features_reach_estimator(records):
    model = pipeline(LogisticRegression(max_iter=2000)).fit(records, (records.Age > 50).astype(int))
    names = model.named_steps["prep"].get_feature_names_out()
    for name in ["age_chol", "age_bp", "chol_fbs", "oldpeak_hr"]:
        assert any(name in feature for feature in names)


def test_imputer_does_not_learn_from_inference(records):
    train = records.iloc[:60].copy()
    train.loc[0, "Cholesterol"] = np.nan
    model = pipeline(LogisticRegression(max_iter=2000)).fit(train, (train.Age > 50).astype(int))
    imputer = model.named_steps["prep"].named_transformers_["numeric"].named_steps["impute"]
    before = imputer.statistics_.copy()
    held_out = records.iloc[60:].copy()
    held_out["Cholesterol"] = 999
    model.predict_proba(held_out)
    np.testing.assert_array_equal(before, imputer.statistics_)
    assert before[2] == train.Cholesterol.median()


def test_zero_sentinel_and_missing_interactions(records):
    records.loc[0, "Cholesterol"] = 0
    transformed = ClinicalFeatures().fit_transform(records)
    assert np.isnan(transformed.loc[0, "Cholesterol"])
    assert np.isnan(transformed.loc[0, "age_chol"])


@pytest.mark.parametrize(
    "column,value", [("Age", -5), ("FastingBS", 0.5), ("Sex", "unknown"), ("Oldpeak", np.inf)]
)
def test_invalid_values_rejected(records, column, value):
    records[column] = records[column].astype(object)
    records.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_frame(records)


def test_schema_and_missing_values(records):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_frame(records.drop(columns="Age"))
    records.loc[0, "Cholesterol"] = np.nan
    assert validate_frame(records).Cholesterol.isna().sum() == 1
    with pytest.raises(ValueError, match="every input missing"):
        validate_frame(pd.DataFrame([{name: np.nan for name in FEATURES}]))


def test_duplicates_removed_before_splitting(records, tmp_path):
    records["HeartDisease"] = (records.Age > 50).astype(int)
    path = tmp_path / "data.csv"
    pd.concat([records, records.iloc[[0]]]).to_csv(path, index=False)
    result, audit = load_data(path)
    assert len(result) == len(records)
    assert audit["duplicates_removed"] == 1
    conflicting = records.iloc[[0]].copy()
    conflicting["HeartDisease"] = 1 - conflicting.HeartDisease
    pd.concat([records, conflicting]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        load_data(path)


def test_threshold_meets_development_target():
    y = [0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.6, 0.4, 0.5, 0.8, 0.9]
    threshold = choose_threshold(y, p, 0.75)
    assert metrics(y, p, threshold)["sensitivity"] >= 0.75


def test_single_class_metrics_are_undefined_not_invented():
    result = metrics([0, 0], [0.1, 0.2])
    assert result["sensitivity"] is None
    assert result["roc_auc"] is None


def test_calibrated_ensemble_clones_and_predicts_missing(records):
    model, _ = candidates(trees=5)["calibrated_vote"]
    y = (records.Age > 50).astype(int)
    probabilities = cross_val_predict(model, records, y, cv=StratifiedKFold(2), method="predict_proba")
    assert probabilities.shape == (len(records), 2)
    assert np.isfinite(probabilities).all()
    model.fit(records, y)
    masked = records.iloc[:2].copy()
    masked["ST_Slope"] = np.nan
    masked["Cholesterol"] = np.nan
    assert np.isfinite(model.predict_proba(masked)).all()


def test_shap_explains_exact_pipeline_probability(records):
    model = pipeline(LogisticRegression(max_iter=2000)).fit(records, (records.Age > 50).astype(int))
    bundle = {"model": model, "background": records.iloc[:5]}
    record = records.iloc[[10]]
    explanation = explain_records(bundle, record, samples=32)
    assert explanation["values"].shape == (1, 11)
    np.testing.assert_allclose(
        explanation["base_value"] + explanation["values"].sum(axis=1),
        model.predict_proba(record)[:, 1],
        atol=1e-5,
    )
    assert decode(encode(record))["Sex"].iloc[0] == record.Sex.iloc[0]

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from heart_research.ecg_risk import weighted_metrics
from heart_research.ecg_service import DEFAULTS, load, predict

ROOT = Path(__file__).parents[1]
RUN = ROOT / "artifacts" / "ecg-risk-v3"
pytestmark = pytest.mark.skipif(not (RUN / "bundle_manifest.json").exists(), reason="ECG run absent")


def test_weighted_results_recompute():
    for name in ["routine", "routine_ecg"]:
        rows = pd.read_csv(RUN / name / "test_predictions.csv")
        result = json.loads((RUN / name / "test_metrics.json").read_text())
        actual = weighted_metrics(rows.target, rows.probability, rows.weight, result["threshold"])
        for key in ["roc_auc", "accuracy", "brier", "sensitivity", "specificity"]:
            np.testing.assert_allclose(actual[key], result["weighted_screening"][key])


def test_reloaded_model_preserves_endpoint():
    model, _ = load(RUN, "routine_ecg")
    result = predict(model, DEFAULTS, "routine_ecg")
    assert result["endpoint"] == "10-year death from heart or cerebrovascular disease"
    assert result["clinically_validated"] is False


def test_ecg_app_and_explanation():
    app = AppTest.from_file(str(ROOT / "ecg_app.py")).run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert "mortality" in app.metric[0].label
    from heart_research.risk_explain import explain_probability
    model, _ = load(RUN, "routine_ecg")
    background = pd.read_csv(RUN / "routine_ecg" / "explanation_background.csv")
    explanation = explain_probability(model, pd.DataFrame([DEFAULTS]), background, samples=64)
    assert explanation["additivity_error"] < 1e-5

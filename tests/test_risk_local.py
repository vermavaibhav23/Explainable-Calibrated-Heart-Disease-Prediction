"""Integration checks for a completed local prospective-risk run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from heart_research.metrics import metrics
from heart_research.risk_service import load_bundle

ROOT = Path(__file__).parents[1]
RUN = ROOT / "artifacts" / "affordable-risk-v2"
pytestmark = pytest.mark.skipif(not (RUN / "bundle_manifest.json").exists(), reason="Local risk run absent")


def test_metrics_recompute_and_holdout_is_disjoint():
    protocol = json.loads((RUN / "protocol.json").read_text())
    splits = protocol["splits"]
    assert not set(splits["test"]) & (set(splits["train"]) | set(splits["calibration"]))
    for name in ["no_lab", "basic_lab"]:
        rows = pd.read_csv(RUN / name / "holdout_predictions.csv")
        result = json.loads((RUN / name / "holdout_metrics.json").read_text())
        assert rows.row.is_unique and set(rows.row) == set(splits["test"])
        recomputed = metrics(rows.y, rows.probability, result["threshold"])
        for key in ["accuracy", "sensitivity", "specificity", "roc_auc", "brier"]:
            np.testing.assert_allclose(recomputed[key], result["screening"][key])


def test_reload_models_and_detect_tampering(tmp_path):
    import shutil
    model, manifest = load_bundle(RUN, "no_lab")
    record = json.loads((ROOT / "examples" / "risk_request.json").read_text())["measurements"]
    assert model.predict_proba(pd.DataFrame([record])).shape == (1, 2)
    (tmp_path / "no_lab").mkdir()
    shutil.copy(RUN / "bundle_manifest.json", tmp_path / "bundle_manifest.json")
    (tmp_path / "no_lab" / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path, "no_lab")


def test_risk_app_both_feature_sets():
    app = AppTest.from_file(str(ROOT / "risk_app.py")).run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert "probability estimate" in app.metric[0].label
    app.radio[0].set_value("Cholesterol and glucose available").run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert not app.exception


def test_explanation_matches_final_probability():
    from heart_research.risk_explain import explain_probability
    model, _ = load_bundle(RUN, "no_lab")
    record = json.loads((ROOT / "examples" / "risk_request.json").read_text())["measurements"]
    background = pd.read_csv(RUN / "no_lab" / "explanation_background.csv")
    result = explain_probability(model, pd.DataFrame([record]), background, samples=64)
    assert result["additivity_error"] < 1e-5
    assert len(result["contributions"]) == len(model.features)

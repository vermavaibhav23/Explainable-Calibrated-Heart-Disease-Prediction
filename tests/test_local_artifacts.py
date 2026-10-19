"""Optional integration tests against a locally completed full experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from heart_research.evaluation import external_evaluate
from heart_research.metrics import metrics

ROOT = Path(__file__).parents[1]
RUN = ROOT / "artifacts" / "full"
pytestmark = pytest.mark.skipif(not (RUN / "manifest.json").exists(), reason="Full local run not present")


def test_saved_metrics_recompute_from_unique_oof_predictions():
    predictions = pd.read_csv(RUN / "oof_predictions.csv")
    comparison = pd.read_csv(RUN / "model_comparison.csv").set_index("model")
    for name, rows in predictions.groupby("model"):
        assert rows.row_id.is_unique
        assert rows.fold.nunique() == 5
        result = metrics(rows.y, rows.probability, rows.threshold)
        for key in ["accuracy", "sensitivity", "specificity", "roc_auc", "brier"]:
            np.testing.assert_allclose(comparison.loc[name, key], result[key])


def test_external_evaluation_rejects_training_overlap(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        external_evaluate(
            RUN / "model.joblib", ROOT / "data" / "raw" / "heart.csv", tmp_path / "external.json"
        )


def test_trained_app_prediction_and_all_pages(monkeypatch):
    monkeypatch.setenv("HEART_RUN_DIR", str(RUN))
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    assert not app.exception
    app.checkbox[0].uncheck()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.metric[0].label == "Model probability"
    for page in ["Validation evidence", "Batch prediction", "About the study"]:
        app.radio[0].set_value(page).run(timeout=60)
        assert not app.exception

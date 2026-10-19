from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_without_model_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("HEART_RUN_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert "Train the model" in app.info[0].value

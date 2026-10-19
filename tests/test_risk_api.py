import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from heart_research.risk_api import app


class DummyModel:
    threshold = 0.15

    def predict_proba(self, X):
        return np.array([[0.9, 0.1]])


@pytest.fixture
def client(monkeypatch):
    # Override only loading to test actual lifespan/auth/validation/response paths.
    monkeypatch.setenv("HEARTLENS_API_KEY", "x" * 32)
    monkeypatch.setattr("heart_research.risk_api.load_bundle", lambda *args: (DummyModel(), {"version": "test"}))
    with TestClient(app) as c:
        yield c


def request():
    from pathlib import Path
    return json.loads((Path(__file__).parents[1] / "examples/risk_request.json").read_text())


def test_requires_authentication(client):
    assert client.post("/predict", json=request()).status_code == 401
    assert client.post("/predict", json=request(), headers={"X-API-Key": "incorrect"}).status_code == 401


def test_response_keeps_endpoint_and_research_status(client):
    response = client.post("/predict", json=request(), headers={"X-API-Key": "x" * 32})
    assert response.status_code == 200
    assert response.json()["endpoint"] == "10-year coronary heart disease"
    assert response.json()["clinically_validated"] is False
    assert response.json()["probability"] == 0.1


def test_invalid_measurements_refused(client):
    record = request()
    record["measurements"]["TenYearCHD"] = 1
    response = client.post("/predict", json=record, headers={"X-API-Key": "x" * 32})
    assert response.status_code == 422


def test_startup_refuses_missing_secret(monkeypatch):
    monkeypatch.delenv("HEARTLENS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="HEARTLENS_API_KEY"), TestClient(app):
        pass


def test_cvd_endpoint_labels_and_authentication(client):
    from heart_research.ecg_service import DEFAULTS
    app.state.ecg_bundles = {"routine_ecg": (DummyModel(), {"version": "test"})}
    body = {"feature_set": "routine_ecg", "measurements": DEFAULTS}
    assert client.post("/predict-cvd-mortality", json=body).status_code == 401
    response = client.post("/predict-cvd-mortality", json=body, headers={"X-API-Key": "x" * 32})
    assert response.status_code == 200
    assert "death" in response.json()["endpoint"]


def test_unavailable_cvd_model_fails_closed(client):
    from heart_research.ecg_service import DEFAULTS
    app.state.ecg_bundles = {}
    response = client.post("/predict-cvd-mortality", json={"measurements": DEFAULTS}, headers={"X-API-Key": "x" * 32})
    assert response.status_code == 503


def test_authentication_precedes_json_parsing(client):
    response = client.post("/predict", content=b"not JSON")
    assert response.status_code == 401


def test_oversized_request_rejected(client):
    response = client.post("/predict", content=b" " * 16385,
                           headers={"X-API-Key": "x" * 32, "Content-Type": "application/json"})
    assert response.status_code == 413


def test_validation_does_not_echo_medical_payload(client):
    record = request()
    record["measurements"]["age"] = "private-medical-input"
    response = client.post("/predict", json=record, headers={"X-API-Key": "x" * 32})
    assert response.status_code == 422
    assert "private-medical-input" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert len(response.headers["x-request-id"]) == 32


def test_readiness_requires_loaded_model(client):
    assert client.get("/ready").status_code == 200
    app.state.bundles = {}
    app.state.ecg_bundles = {}
    assert client.get("/ready").status_code == 503


def test_cvd_only_deployment_does_not_load_chd(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTLENS_API_KEY", "x" * 32)
    monkeypatch.setenv("HEARTLENS_MODELS", "cvd")
    monkeypatch.setenv("HEARTLENS_ECG_ARTIFACTS", str(tmp_path))
    (tmp_path / "bundle_manifest.json").write_text("{}")
    monkeypatch.setattr("heart_research.risk_api.load_ecg", lambda *args: (DummyModel(), {"version": "test"}))
    with TestClient(app) as c:
        assert c.get("/ready").status_code == 200
        assert not app.state.bundles
        assert c.post("/predict", json=request(), headers={"X-API-Key": "x" * 32}).status_code == 503


def test_empty_deployment_refuses_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTLENS_API_KEY", "x" * 32)
    monkeypatch.setenv("HEARTLENS_MODELS", "cvd")
    monkeypatch.setenv("HEARTLENS_ECG_ARTIFACTS", str(tmp_path))
    with pytest.raises(RuntimeError, match="No model bundle"), TestClient(app):
        pass

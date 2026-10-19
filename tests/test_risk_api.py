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

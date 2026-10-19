"""Authenticated local research API. Set HEARTLENS_API_KEY before starting."""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from .risk_service import load_bundle, predict_record

key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app):
    token = os.environ.get("HEARTLENS_API_KEY", "")
    if len(token) < 32:
        raise RuntimeError("Set HEARTLENS_API_KEY to a randomly generated secret of at least 32 characters.")
    root = Path(os.environ.get("HEARTLENS_RISK_ARTIFACTS", "artifacts/affordable-risk-v2"))
    app.state.token = token
    app.state.bundles = {name: load_bundle(root, name) for name in ["no_lab", "basic_lab"]}
    yield


app = FastAPI(title="HeartLens affordable CHD research API", version="2.0.0", lifespan=lifespan)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    feature_set: Literal["no_lab", "basic_lab"] = "no_lab"
    measurements: dict[str, float | int | None] = Field(max_length=14)


@app.get("/health")
def health():
    return {"status": "ready", "research_only": True}


@app.post("/predict")
def predict(request: Request, key: str | None = Security(key_header)):
    if not key or not secrets.compare_digest(key, app.state.token):
        raise HTTPException(401, "Invalid API key")
    model, manifest = app.state.bundles[request.feature_set]
    try:
        return predict_record(model, request.measurements, request.feature_set, manifest["version"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

"""Validated inference for the affordable CHD research models."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .risk import FEATURE_SETS

BINARY = {"male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"}
RANGES = {"age": (32, 70), "cigsPerDay": (0, 100), "sysBP": (70, 300),
          "diaBP": (40, 180), "BMI": (12, 70), "heartRate": (30, 220),
          "totChol": (80, 750), "glucose": (30, 600)}
CORE = {"male", "age", "currentSmoker", "sysBP", "diaBP", "BMI"}


def validate_record(record, feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown feature set.")
    features = FEATURE_SETS[feature_set]
    if set(record) - set(features):
        raise ValueError("Unexpected fields: " + ", ".join(sorted(set(record) - set(features))))
    missing = [f for f in features if record.get(f) is None]
    if CORE.intersection(missing):
        raise ValueError("Required measurements missing: " + ", ".join(sorted(CORE.intersection(missing))))
    if len(missing) / len(features) > 0.3:
        raise ValueError("Too many missing inputs; provide at least 70% of measurements.")
    clean = {}
    for name in features:
        value = record.get(name)
        if value is None:
            clean[name] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric value or null.")
        if name in BINARY and value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1.")
        if name in RANGES and not RANGES[name][0] <= value <= RANGES[name][1]:
            raise ValueError(f"{name} is outside the supported range {RANGES[name]}.")
        clean[name] = float(value)
    if clean["sysBP"] <= clean["diaBP"]:
        raise ValueError("Systolic pressure must exceed diastolic pressure.")
    if clean["currentSmoker"] == 0 and clean.get("cigsPerDay", 0) > 0:
        raise ValueError("Non-smoker cannot have positive current cigarettes per day.")
    return pd.DataFrame([clean], columns=features), missing


def load_bundle(root, feature_set):
    """Only load administrator-provided local artifacts; never accept uploaded pickle files."""
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown feature set.")
    root = Path(root)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    path = root / feature_set / "model.joblib"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest["model_sha256"][feature_set]:
        raise ValueError("Artifact checksum mismatch; inference refused.")
    model = joblib.load(path)
    if model.features != FEATURE_SETS[feature_set]:
        raise ValueError("Artifact schema mismatch.")
    return model, manifest


def predict_record(model, record, feature_set, version):
    X, missing = validate_record(record, feature_set)
    probability = float(model.predict_proba(X)[0, 1])
    return {"model_version": version, "endpoint": "10-year coronary heart disease",
            "research_only": True, "clinically_validated": False,
            "feature_set": feature_set, "probability": probability,
            "research_screen_positive": probability >= model.threshold,
            "research_threshold": model.threshold, "imputed_fields": missing,
            "limitations": "Historical teaching data; not MI-only risk, acute diagnosis or treatment advice."}

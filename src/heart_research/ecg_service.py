"""Schema and endpoint contract for the NHANES CVD-mortality research model."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .nhanes import ENDPOINT, SETS

BOUNDS = {"age": (40, 89), "systolic": (70, 300), "diastolic": (10, 180), "bmi": (12, 70),
          "total_cholesterol": (50, 750), "hdl": (5, 220), "hba1c": (2, 20), "creatinine": (0.05, 20),
          "ecg_rate": (30, 220), "pr_ms": (20, 500), "qrs_ms": (40, 250), "qt_ms": (150, 700),
          "p_axis": (-180, 180), "qrs_axis": (-180, 180), "t_axis": (-180, 180)}
BINARY = {"male", "smoking", "diabetes", "bp_medication", "prior_mi", "major_ecg_abnormality"}
DEFAULTS = dict(age=60, male=0, systolic=130, diastolic=80, bmi=25, smoking=0, diabetes=0,
                bp_medication=0, prior_mi=0, total_cholesterol=200, hdl=50, hba1c=5.5,
                creatinine=0.9, ecg_rate=72, pr_ms=160, qrs_ms=90, qt_ms=400,
                p_axis=45, qrs_axis=30, t_axis=45, major_ecg_abnormality=0)
LABELS = {"age": "Age (years)", "male": "Male sex recorded in source data", "systolic": "Systolic BP (mmHg)",
          "diastolic": "Diastolic BP (mmHg)", "bmi": "BMI (kg/m²)", "smoking": "Current smoking",
          "diabetes": "Known diabetes", "bp_medication": "BP medication", "prior_mi": "Previous heart attack",
          "total_cholesterol": "Total cholesterol (mg/dL)", "hdl": "HDL cholesterol (mg/dL)",
          "hba1c": "HbA1c (%)", "creatinine": "Standardized creatinine (mg/dL)",
          "ecg_rate": "ECG heart rate (beats/min)", "pr_ms": "PR interval (ms)", "qrs_ms": "QRS interval (ms)",
          "qt_ms": "QT interval, not QTc (ms)", "p_axis": "P axis (degrees)",
          "qrs_axis": "QRS axis (degrees)", "t_axis": "T axis (degrees)",
          "major_ecg_abnormality": "Major ECG abnormality (Minnesota criteria)"}


def validate(record, feature_set):
    if feature_set not in SETS:
        raise ValueError("Unknown feature set.")
    features = SETS[feature_set]
    if set(record) - set(features):
        raise ValueError("Unexpected input fields.")
    required = {"age", "male", "systolic", "diastolic", "bmi"}
    if feature_set == "routine_ecg":
        required.update(["ecg_rate", "pr_ms", "qrs_ms", "qt_ms"])
    missing = [f for f in features if record.get(f) is None]
    if required.intersection(missing) or len(missing) / len(features) > 0.3:
        raise ValueError("Required measurements are missing or more than 30% of inputs are unknown.")
    cleaned = {}
    for feature in features:
        value = record.get(feature)
        if value is None:
            cleaned[feature] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number or null.")
        if feature in BINARY and value not in (0, 1):
            raise ValueError(f"{feature} must be 0 or 1.")
        if feature in BOUNDS and not BOUNDS[feature][0] <= value <= BOUNDS[feature][1]:
            raise ValueError(f"{feature} outside supported range {BOUNDS[feature]}.")
        cleaned[feature] = float(value)
    if cleaned["systolic"] <= cleaned["diastolic"]:
        raise ValueError("Systolic pressure must exceed diastolic pressure.")
    return pd.DataFrame([cleaned], columns=features), missing


def load(root, feature_set):
    if feature_set not in SETS:
        raise ValueError("Unknown feature set.")
    root = Path(root)
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    path = root / feature_set / "model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"][feature_set]:
        raise ValueError("Model checksum mismatch.")
    model = joblib.load(path)
    if model.features != SETS[feature_set]:
        raise ValueError("Model schema mismatch.")
    return model, manifest


def predict(model, record, feature_set):
    X, missing = validate(record, feature_set)
    p = float(model.predict_proba(X)[0, 1])
    return {"endpoint": ENDPOINT, "probability": p, "research_threshold": model.threshold,
            "research_screen_positive": p >= model.threshold, "imputed_fields": missing,
            "research_only": True, "clinically_validated": False,
            "limitations": "Historical US ages 40–89; fatal outcomes only; not acute diagnosis or validated personal risk."}

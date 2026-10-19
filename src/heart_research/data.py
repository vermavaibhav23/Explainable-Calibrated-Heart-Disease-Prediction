"""Strict raw schema, audit, and reproducible data acquisition."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORIES = {
    "Sex": ["F", "M"],
    "ChestPainType": ["TA", "ATA", "NAP", "ASY"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Up", "Flat", "Down"],
}
FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
TARGET = "HeartDisease"
# Broad schema limits, not diagnostic cutoffs.
BOUNDS = {
    "Age": (18, 120),
    "RestingBP": (0, 300),
    "Cholesterol": (0, 1000),
    "FastingBS": (0, 1),
    "MaxHR": (20, 260),
    "Oldpeak": (-10, 15),
}
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction"
DATA_PAGE = "https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction"


def validate_frame(frame: pd.DataFrame, target: bool = False) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if target else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The input contains no records.")
    out = frame[required].copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        lo, hi = BOUNDS[col]
        bad = out[col].notna() & (~np.isfinite(out[col]) | ~out[col].between(lo, hi))
        if bad.any():
            raise ValueError(f"{col} must be missing or between {lo} and {hi}.")
    if not out.FastingBS.dropna().isin([0, 1]).all():
        raise ValueError("FastingBS must be 0 or 1.")
    for col, allowed in CATEGORIES.items():
        out[col] = out[col].replace(r"^\s*$", np.nan, regex=True)
        bad = out[col].notna() & ~out[col].isin(allowed)
        if bad.any():
            raise ValueError(f"Unknown {col} category; allowed: {allowed}")
    if out[FEATURES].isna().all(axis=1).any():
        raise ValueError("A record cannot have every input missing.")
    if target:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="raise")
        if out[TARGET].isna().any() or not out[TARGET].isin([0, 1]).all():
            raise ValueError("HeartDisease labels must be non-missing 0 or 1.")
        out[TARGET] = out[TARGET].astype(int)
    return out


def load_data(path):
    raw = pd.read_csv(path)
    df = validate_frame(raw, target=True)
    # Treat zero BP/cholesterol as missing before duplicate checks and splitting.
    zero_counts = {c: int((df[c] == 0).sum()) for c in ["RestingBP", "Cholesterol"]}
    df[["RestingBP", "Cholesterol"]] = df[["RestingBP", "Cholesterol"]].replace(0, np.nan)
    contradictory = df.groupby(FEATURES, dropna=False)[TARGET].nunique().gt(1)
    if contradictory.any():
        raise ValueError("Identical feature records have conflicting labels; resolve before training.")
    duplicates = int(df.duplicated(subset=FEATURES).sum())
    df = df.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Training requires both outcome classes.")
    audit = {
        "source_rows": len(raw),
        "retained_rows": len(df),
        "duplicates_removed": duplicates,
        "zero_values_treated_as_missing": zero_counts,
        "missing_counts": df.isna().sum().to_dict(),
        "class_counts": df[TARGET].value_counts().to_dict(),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return df, audit


def download_data(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; use it or choose another path.")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    payload = archive.read("heart.csv")
    validate_frame(pd.read_csv(io.BytesIO(payload)), target=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "source": DATA_PAGE,
                "download_url": DATA_URL,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "note": "Upstream dataset terms apply. Data are not included in the source repository.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination

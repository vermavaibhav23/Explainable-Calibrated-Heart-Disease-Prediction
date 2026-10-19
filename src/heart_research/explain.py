"""SHAP attributions for the exact deployed end-to-end probability function."""

import numpy as np
import pandas as pd

from .data import CATEGORIES, FEATURES, NUMERIC, validate_frame


def encode(frame):
    result = frame[FEATURES].copy()
    for col, categories in CATEGORIES.items():
        result[col] = result[col].map({value: i for i, value in enumerate(categories)})
    return result.astype(float).to_numpy()


def decode(values):
    frame = pd.DataFrame(values, columns=FEATURES)
    for col, categories in CATEGORIES.items():
        frame[col] = frame[col].map({i: value for i, value in enumerate(categories)})
    for col in NUMERIC:
        frame[col] = frame[col].astype(float)
    return frame


def explain_records(bundle, frame, samples=128):
    import shap

    frame = validate_frame(frame)
    background = encode(bundle["background"])

    def predict(values):
        return bundle["model"].predict_proba(decode(values))[:, 1]

    # Kernel SHAP masks entire original columns, then recomputes interactions in
    # the actual fitted pipeline. Category codes are only an adapter, not an ordinal model.
    state = np.random.get_state()
    try:
        np.random.seed(42)
        explainer = shap.KernelExplainer(predict, background, link="identity")
        values = np.asarray(explainer.shap_values(encode(frame), nsamples=samples, l1_reg=0.0, silent=True))
    finally:
        np.random.set_state(state)
    probability = predict(encode(frame))
    base = float(explainer.expected_value)
    if not np.allclose(base + values.sum(axis=1), probability, atol=1e-5):
        raise RuntimeError("SHAP additivity verification failed.")
    return {"values": values, "base_value": base, "probabilities": probability, "features": FEATURES}

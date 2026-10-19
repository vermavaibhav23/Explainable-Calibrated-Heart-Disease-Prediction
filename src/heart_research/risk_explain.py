"""Explain the final calibrated probability, not an unrelated base learner."""

import numpy as np
import pandas as pd
import shap


def explain_probability(model, record, background, samples=256):
    features = model.features
    background = background[features].head(30)

    def probability(values):
        return model.predict_proba(pd.DataFrame(values, columns=features))[:, 1]

    explainer = shap.KernelExplainer(probability, background)
    # SHAP's coalition sample uses global NumPy RNG; preserve caller state.
    state = np.random.get_state()
    try:
        np.random.seed(2026)
        values = np.asarray(explainer.shap_values(record[features], nsamples=samples, silent=True)).reshape(-1)
    finally:
        np.random.set_state(state)
    base = float(explainer.expected_value)
    predicted = float(probability(record[features].to_numpy())[0])
    return {"baseline_probability": base, "probability": predicted,
            "additivity_error": float(abs(base + values.sum() - predicted)),
            "contributions": pd.DataFrame({"feature": features, "probability_contribution": values})}

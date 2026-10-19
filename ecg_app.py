"""Routine checkup + ECG cardiovascular mortality research interface."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from heart_research.ecg_service import BINARY, BOUNDS, DEFAULTS, LABELS, load, predict, validate
from heart_research.nhanes import SETS

st.set_page_config(page_title="HeartLens | ECG risk research", page_icon="♥", layout="wide")
st.title("HeartLens · Routine checkup + ECG")
st.write("10-year risk of death from heart or cerebrovascular disease · NHANES III research model")
st.warning("Historical-cohort research demonstration. This is not a heart-attack diagnosis or a clinically validated personal risk estimate.")
root = Path("artifacts/ecg-risk-v3")
if not (root / "bundle_manifest.json").exists():
    st.info("The ECG experiment has not been packaged yet. See docs/ECG_RISK.md.")
    st.stop()


@st.cache_resource
def bundle(name):
    return load(root, name)


feature_set = st.radio("Available measurements", ["routine_ecg", "routine"],
                        format_func=lambda x: "Routine checkup + resting ECG" if x == "routine_ecg" else "Routine checkup only",
                        horizontal=True)
model, manifest = bundle(feature_set)
st.caption("A fictional example is prefilled. Replace values with measurements. ECG intervals/axes come from a standardized ECG report; do not guess them.")
with st.form("measurements"):
    columns = st.columns(3)
    record = {}
    for i, feature in enumerate(SETS[feature_set]):
        with columns[i % 3]:
            if feature in BINARY:
                choice = st.selectbox(LABELS[feature], ["No", "Yes", "Unknown"], key=feature)
                record[feature] = {"No": 0, "Yes": 1, "Unknown": None}[choice]
            else:
                low, high = BOUNDS[feature]
                record[feature] = st.number_input(LABELS[feature], float(low), float(high), float(DEFAULTS[feature]), key=feature)
    explain = st.checkbox("Explain this prediction")
    submitted = st.form_submit_button("Calculate research estimate")
if submitted:
    try:
        result = predict(model, record, feature_set)
        st.metric("Estimated 10-year cardiovascular mortality", f"{result['probability']:.1%}")
        st.caption("Includes fatal heart and cerebrovascular disease; does not count nonfatal heart attacks or strokes. This output does not determine treatment.")
        st.json(result)
        if explain:
            from heart_research.risk_explain import explain_probability
            with st.spinner("Explaining the complete probability model…"):
                X, _ = validate(record, feature_set)
                explanation = explain_probability(model, X, pd.read_csv(root / feature_set / "explanation_background.csv"))
            st.bar_chart(explanation["contributions"].set_index("feature"))
            st.caption("Approximate SHAP effects on the model probability; associations are not causes or recommended interventions.")
    except ValueError as exc:
        st.error(str(exc))
st.subheader("Later-phase test evidence")
scores = json.loads((root / feature_set / "test_metrics.json").read_text())
columns = st.columns(4)
for column, key in zip(columns, ["roc_auc", "sensitivity", "specificity", "precision"], strict=True):
    column.metric(key.replace("_", " ").title(), f"{scores['weighted_screening'][key]:.3f}")
st.caption("Survey-weighted performance on the held-out 1991–1994 phase. Earlier-phase data alone determined model, calibration and screening threshold.")
st.image(str(root / feature_set / "evaluation.png"))
with st.expander("Model selection and limitations"):
    st.json(scores)
    st.dataframe(pd.DataFrame(json.loads((root / feature_set / "development_search.json").read_text())), use_container_width=True)
    st.write("Development scores are selection-biased. Temporal evaluation is not an independent modern hospital validation, and public mortality times/causes are perturbed for some records.")

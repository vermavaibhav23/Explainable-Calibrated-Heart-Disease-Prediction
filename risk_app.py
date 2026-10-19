"""VS Code/Streamlit interface for the separate prospective CHD benchmark."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from heart_research.risk_service import load_bundle, predict_record, validate_record

st.set_page_config(page_title="HeartLens | Affordable risk", page_icon="♥", layout="wide")
st.title("HeartLens · Affordable risk research")
st.write("Explore a 10-year coronary heart disease model using everyday measurements.")
st.warning("Research demonstration: this model does not predict heart attacks specifically and has not been clinically validated.")
root = Path("artifacts/affordable-risk-v2")
if not (root / "bundle_manifest.json").exists():
    st.info("Train and package the risk benchmark first; see docs/AFFORDABLE_RISK.md.")
    st.stop()


@st.cache_resource
def bundle(name):
    return load_bundle(root, name)


label = st.radio("Measurements available", ["No blood tests", "Cholesterol and glucose available"], horizontal=True)
name = "no_lab" if label == "No blood tests" else "basic_lab"
model, manifest = bundle(name)
with st.form("risk"):
    left, middle, right = st.columns(3)
    with left:
        age = st.number_input("Age (supported: 32–70)", 32, 70, 50)
        male = st.selectbox("Sex recorded in source data", ["Female", "Male"])
        smoker = st.checkbox("Currently smokes")
        cigarettes = st.number_input("Cigarettes per day", 0, 100, 0)
    with middle:
        systolic = st.number_input("Systolic BP (mmHg)", 70.0, 300.0, 120.0)
        diastolic = st.number_input("Diastolic BP (mmHg)", 40.0, 180.0, 80.0)
        bmi = st.number_input("BMI (kg/m²)", 12.0, 70.0, 24.0)
        pulse = st.number_input("Resting heart rate (beats/min)", 30.0, 220.0, 72.0)
    with right:
        bp_meds = st.checkbox("Takes BP medication")
        hypertension = st.checkbox("Known hypertension")
        diabetes = st.checkbox("Known diabetes")
        stroke = st.checkbox("Previous stroke")
        if name == "basic_lab":
            cholesterol = st.number_input("Total cholesterol (mg/dL)", 80.0, 750.0, 200.0)
            glucose = st.number_input("Glucose (mg/dL)", 30.0, 600.0, 90.0)
    explain = st.checkbox("Explain which inputs influenced this model estimate")
    submitted = st.form_submit_button("Calculate research estimate")
if submitted:
    record = dict(age=age, male=int(male == "Male"), currentSmoker=int(smoker), cigsPerDay=cigarettes,
                  sysBP=systolic, diaBP=diastolic, BMI=bmi, heartRate=pulse, BPMeds=int(bp_meds),
                  prevalentHyp=int(hypertension), diabetes=int(diabetes), prevalentStroke=int(stroke))
    if name == "basic_lab":
        record.update(totChol=cholesterol, glucose=glucose)
    try:
        result = predict_record(model, record, name, manifest["version"])
        st.metric("Model's 10-year CHD probability estimate", f"{result['probability']:.1%}")
        st.caption("This percentage is a model estimate on historical data, not a verified personal risk or a treatment recommendation.")
        st.json(result)
        if explain:
            from heart_research.risk_explain import explain_probability
            X, _ = validate_record(record, name)
            background = pd.read_csv(root / name / "explanation_background.csv")
            with st.spinner("Explaining the final calibrated model…"):
                explanation = explain_probability(model, X, background)
            st.bar_chart(explanation["contributions"].set_index("feature"))
            st.caption("Approximate SHAP contributions relative to a development-data background. These explain model behaviour, not causes or the effect of changing a habit.")
    except ValueError as exc:
        st.error(str(exc))
st.subheader("Held-out evidence")
result = json.loads((root / name / "holdout_metrics.json").read_text())
cols = st.columns(4)
for col, key, title in zip(cols, ["roc_auc", "sensitivity", "specificity", "precision"],
                            ["AUROC", "Sensitivity", "Specificity", "Precision"], strict=True):
    col.metric(title, f"{result['screening'][key]:.3f}")
st.caption("848 held-out records. Screening threshold selected using separate calibration records. No independent population validation.")
st.dataframe(pd.DataFrame(json.loads((root / name / "development_search.json").read_text())), use_container_width=True)
st.caption("Development scores above were used for selection and are optimistic; use the held-out evidence for performance claims.")

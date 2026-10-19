"""Launch from VS Code terminal: python -m streamlit run app.py"""

import io
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from heart_research.data import BOUNDS, CATEGORIES, FEATURES, validate_frame
from heart_research.explain import explain_records
from heart_research.reporting import prediction_report

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get("HEART_RUN_DIR", str(ROOT / "artifacts" / "full")))
st.set_page_config(page_title="HeartLens | Research workspace", page_icon="🫀", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f4f8f7;color:#153b3b}
h1,h2,h3{letter-spacing:-.035em}
[data-testid="stMetric"]{background:white;border:1px solid #dbe9e4;border-radius:16px;padding:18px}
.hero{padding:28px 32px;background:#103e3b;color:#f4fbf8;border-radius:22px;margin-bottom:24px}
.hero h1{color:#f4fbf8;margin:8px 0}.hero p{color:#b8d8cc;max-width:760px}
.eyebrow{font-size:12px;letter-spacing:.2em;text-transform:uppercase;color:#94d7bb}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(path, modified):
    return joblib.load(path)


st.markdown(
    '<div class="hero"><div class="eyebrow">HeartLens / Research workspace</div>'
    "<h1>Understand the prediction.</h1>"
    "<p>Explore calibrated heart disease classification, inspect individual explanations, "
    "and examine the evidence behind the model.</p></div>",
    unsafe_allow_html=True,
)
st.caption(
    "RESEARCH PROTOTYPE · Internal validation only · Not a diagnosis or a future heart-attack risk score"
)
model_path = RUN / "model.joblib"
if not model_path.exists():
    st.info("Train the model once to open the research workspace.")
    st.code("python -m heart_research.cli download\npython -m heart_research.cli train --jobs 2")
    st.stop()
bundle = load_bundle(str(model_path), model_path.stat().st_mtime)
with st.sidebar:
    st.title("HeartLens")
    page = st.radio(
        "Workspace", ["Individual prediction", "Validation evidence", "Batch prediction", "About the study"]
    )
    st.divider()
    st.caption(f"Model: calibrated voting ensemble\n\nTraining records: {bundle['manifest']['rows']}")
    st.caption(f"Decision threshold: {bundle['threshold']:.2f}")
    if bundle["manifest"]["smoke_run"]:
        st.warning("Smoke-test model. Run the full protocol for research results.")

LABELS = {
    "Age": "Age (years)",
    "Sex": "Sex (dataset category)",
    "ChestPainType": "Chest pain type",
    "RestingBP": "Resting blood pressure (mm Hg)",
    "Cholesterol": "Cholesterol (mg/dL)",
    "FastingBS": "Fasting blood sugar >120 mg/dL",
    "RestingECG": "Resting ECG category",
    "MaxHR": "Maximum heart rate (bpm)",
    "ExerciseAngina": "Exercise-induced angina",
    "Oldpeak": "Oldpeak (ST depression)",
    "ST_Slope": "Exercise ST slope",
}
DISPLAY = {
    "TA": "TA · Typical angina",
    "ATA": "ATA · Atypical angina",
    "NAP": "NAP · Non-anginal pain",
    "ASY": "ASY · Asymptomatic",
    "LVH": "LVH · Left ventricular hypertrophy",
    "ST": "ST · ST–T abnormality",
    "F": "Female",
    "M": "Male",
    "Y": "Yes",
    "N": "No",
}
DEFAULTS = {
    "Age": 52.0,
    "RestingBP": 130.0,
    "Cholesterol": 220.0,
    "FastingBS": 0.0,
    "MaxHR": 150.0,
    "Oldpeak": 1.0,
}


def show_prediction(record, explanation_requested):
    frame = validate_frame(record)
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    threshold = bundle["threshold"]
    missing_count = int(frame.isna().sum().sum())
    a, b, c = st.columns(3)
    a.metric("Model probability", f"{probability:.1%}")
    b.metric("Model classification", "Positive" if probability >= threshold else "Negative")
    c.metric("Missing inputs", missing_count)
    if missing_count:
        st.info(
            "Missing values use training-derived imputation. Check the missing-input experiments before interpreting this output."
        )
    outside = [
        col
        for col, (lo, hi) in bundle["ranges"].items()
        if frame[col].notna().all() and not lo <= frame[col].iloc[0] <= hi
    ]
    if outside:
        st.warning("Outside the training data's observed range: " + ", ".join(outside))
    if abs(probability - threshold) <= 0.05:
        st.info(
            "This prediction is close to the decision threshold. This is a threshold-proximity flag, not a confidence interval."
        )
    explanation = None
    if explanation_requested:
        with st.spinner("Explaining the calibrated ensemble..."):
            explanation = explain_records(bundle, frame)
        values = pd.Series(explanation["values"][0] * 100, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.barh(values.index, values.values, color=["#c06b51" if x > 0 else "#278c82" for x in values])
        ax.axvline(0, color="#456", linewidth=0.8)
        ax.set_xlabel("Contribution to model probability (percentage points)")
        ax.set_title("Why this model produced this probability")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Background probability {explanation['base_value']:.1%} + SHAP contributions = {probability:.1%}. "
            "These are approximate model attributions, not causal effects or treatment recommendations."
        )
    html = prediction_report(frame, probability, threshold, explanation)
    st.download_button("Download research report", html, "heartlens-report.html", "text/html")


if page == "Individual prediction":
    st.subheader("Explore one record")
    st.write("Enter the clinical attributes used by the study. Mark unavailable values as missing.")
    st.caption("The form starts with a fictional example. These inputs include exercise-test measurements.")
    with st.form("clinical_inputs"):
        unavailable = st.multiselect("Unavailable measurements", FEATURES)
        columns = st.columns(3)
        values = {}
        for i, name in enumerate(FEATURES):
            with columns[i % 3]:
                if name in CATEGORIES:
                    values[name] = st.selectbox(
                        LABELS[name], CATEGORIES[name], format_func=lambda v: DISPLAY.get(v, v)
                    )
                elif name == "FastingBS":
                    values[name] = float(
                        st.selectbox(LABELS[name], [0, 1], format_func=lambda v: "Yes" if v else "No")
                    )
                else:
                    lo, hi = BOUNDS[name]
                    # Zero is a missing-value sentinel in the training dataset; the UI uses an explicit missing control.
                    lo = 1 if name in ["RestingBP", "Cholesterol"] else lo
                    values[name] = st.number_input(
                        LABELS[name],
                        min_value=float(lo),
                        max_value=float(hi),
                        value=DEFAULTS[name],
                        step=0.1 if name == "Oldpeak" else 1.0,
                    )
        explanation_requested = st.checkbox("Include an individual SHAP explanation", value=True)
        submitted = st.form_submit_button("Generate model prediction", type="primary")
    if submitted:
        for name in unavailable:
            values[name] = np.nan
        st.session_state["record"] = pd.DataFrame([values])
        st.session_state["explain"] = explanation_requested
    if "record" in st.session_state:
        try:
            show_prediction(st.session_state["record"], st.session_state["explain"])
        except ValueError as error:
            st.error(str(error))

elif page == "Validation evidence":
    st.subheader("Evidence from unseen outer folds")
    st.write(
        "Each prediction here was generated without training on that record. Model selection, calibration, "
        "and threshold selection were confined to the corresponding training folds."
    )
    comparison = pd.read_csv(RUN / "model_comparison.csv")
    row = comparison.set_index("model").loc["calibrated_vote"]
    columns = st.columns(4)
    for col, name in zip(columns, ["roc_auc", "sensitivity", "specificity", "brier"]):
        col.metric(name.replace("_", " ").title(), f"{row[name]:.3f}")
    tabs = st.tabs(["Models", "Calibration & discrimination", "Missing inputs", "Subgroups", "Uncertainty"])
    with tabs[0]:
        st.dataframe(comparison, hide_index=True)
        st.caption(
            "PR-AUC is reported as average precision. Thresholds vary by outer fold. Lower Brier score is better."
        )
    with tabs[1]:
        for filename in ["calibration.png", "roc.png", "precision_recall.png"]:
            st.image(str(RUN / filename))
    with tabs[2]:
        st.dataframe(pd.read_csv(RUN / "robustness.csv"), hide_index=True)
        st.caption(
            "Same outer models and thresholds; selected validation inputs are masked. Simulated missingness only."
        )
    with tabs[3]:
        st.dataframe(pd.read_csv(RUN / "subgroups.csv"), hide_index=True)
        st.caption(
            "Descriptive subgroup checks, not a fairness certification. Interpret small groups cautiously."
        )
    with tabs[4]:
        intervals = json.loads((RUN / "confidence_intervals.json").read_text())["calibrated_vote"]
        st.dataframe(pd.DataFrame(intervals, index=["Lower 95%", "Upper 95%"]).T)
        st.caption(
            "Stratified bootstrap of out-of-fold predictions. These intervals exclude full model-training variability."
        )
    st.download_button(
        "Download complete experiment report",
        (RUN / "report.html").read_bytes(),
        "heartlens-experiment.html",
        "text/html",
    )

elif page == "Batch prediction":
    st.subheader("Explore a CSV of records")
    st.write(
        "Upload the 11 feature columns. Blank cells are treated as missing; unexpected categories are rejected."
    )
    st.download_button(
        "Download fictional input template",
        (ROOT / "examples" / "patients.csv").read_bytes(),
        "patients.csv",
        "text/csv",
    )
    upload = st.file_uploader("Clinical CSV", type="csv")
    if upload is not None:
        try:
            raw = pd.read_csv(io.BytesIO(upload.getvalue()))
            if len(raw) > 5000:
                raise ValueError("Use the CLI for more than 5,000 records.")
            frame = validate_frame(raw)
            result = frame.copy()
            result["model_probability"] = bundle["model"].predict_proba(frame)[:, 1]
            result["model_class"] = (result.model_probability >= bundle["threshold"]).astype(int)
            result["decision_threshold"] = bundle["threshold"]
            st.dataframe(result, hide_index=True)
            st.download_button(
                "Download predictions", result.to_csv(index=False), "predictions.csv", "text/csv"
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))
    st.caption("The local app processes uploads in memory and does not save uploaded CSVs.")

else:
    st.subheader("Explainable and Calibrated Heart Disease Prediction System with Robust Validation")
    st.write(
        "Research question: How do calibration and missing clinical inputs affect the discrimination, "
        "sensitivity, and reliability of ensemble heart disease classification?"
    )
    st.markdown(
        "**Pipeline:** clinical input → feature engineering → fold-fitted preprocessing → "
        "logistic regression / ExtraTrees / SVM voting → sigmoid calibration → development-selected threshold."
    )
    for limitation in bundle["manifest"]["limitations"]:
        st.write("• " + limitation)
    st.markdown(
        "Dataset: [Heart Failure Prediction Dataset, fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). "
        "The modeled target is HeartDisease, not future heart failure or future heart attack."
    )
    st.json(bundle["manifest"])

# Affordable prospective risk benchmark

This extension studies **10-year coronary heart disease (CHD)** using the MIT OpenCourseWare Framingham teaching CSV. It does not establish heart-attack-only risk, diagnose an acute attack, or replace the original existing-disease classifier. There is no claim of clinical readiness or superiority over IEEE papers.

## Run in VS Code / Windows

From this repository's root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-risk.txt
.\.venv\Scripts\python.exe scripts/download_risk_data.py
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m heart_research.risk --data data/raw/framingham.csv --output artifacts/affordable-risk-v2 --jobs 2 --iterations 10
.\.venv\Scripts\python.exe scripts/package_risk.py --data data/raw/framingham.csv --run artifacts/affordable-risk-v2
.\.venv\Scripts\python.exe -m streamlit run risk_app.py
```

Training refuses to overwrite an existing run. Use a new directory for a genuinely new experiment, retain its protocol and report all changes; do not search random seeds for a better test score. The reference run uses seed 2026, chosen before training.

## Inputs and endpoint

The no-lab model uses age, the dataset's binary recorded sex, current smoking, cigarettes/day, known diabetes, known hypertension, BP medication, previous stroke, systolic/diastolic pressure, BMI and resting pulse. History of diabetes is a known diagnosis, not an inferred lab result; a user who does not know must not silently enter “no.” Education is excluded. The basic-lab comparator adds total cholesterol and glucose, both in mg/dL. No ECG, exercise stress test, troponin, imaging or angiography is used.

The source data has 4,240 rows, 644 positive outcomes and ages 32–70. `TenYearCHD` is broader than myocardial infarction. Time-to-event and censoring information are not available in this CSV, so this is fixed-horizon binary prediction, not survival analysis. Participant identifiers and follow-up completeness cannot be independently checked. The history fields do not establish a rigorously disease-free primary-prevention cohort. Generalization to India, other ages, or contemporary clinical practice has not been shown.

MIT provides the [CSV](https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/resources/framingham/) as course material. NHLBI separately warns that its [anonymized teaching datasets](https://biolincc.nhlbi.nih.gov/teaching/) are unsuitable for publication. The exact relationship between the MIT CSV and those altered datasets is unresolved. Treat this benchmark as educational until provenance, permitted uses and outcome definitions are established with the data custodian. Real [Framingham cohort data](https://biolincc.nhlbi.nih.gov/studies/framcohort/) requires an access process; public download availability is not proof of clinical suitability.

## Evaluation protocol

1. Validate numeric schema and labels. Collapse identical baseline profiles; reject conflicting outcomes. Check feature-specific profile overlap when packaging.
2. One preselected stratified split: 60% development, 20% calibration, 20% untouched test (2,544 / 848 / 848 records for this file).
3. Five-fold development search for logistic regression, spline logistic regression, Random Forest, Extra Trees, histogram boosting, CatBoost, XGBoost and LightGBM. Up to ten configurations per family, six for each logistic family. Median imputation, missingness indicators, scaling and spline fitting stay inside folds.
4. Compare the strongest single development model with an equal-weight blend of the top three. Select by development OOF AUROC. These OOF scores reuse the model-selection data and **are not unbiased performance estimates**.
5. On the separate calibration partition, compare no recalibration, sigmoid and isotonic methods using five-fold Brier score. Select a research screening threshold to target at least 80% sensitivity on calibration OOF predictions. The selected threshold is not a clinical treatment recommendation, and its target is not guaranteed in new patients.
6. Freeze the model, calibrator and threshold, then evaluate the test partition once for each predeclared feature set. No retraining on test labels and no further tuning based on these results.
7. Report AUROC, average precision, Brier score, confusion counts, sensitivity, specificity, precision, both screening and 0.5-threshold accuracy, and a constant-prevalence baseline. Include 1,000 stratified bootstrap percentile intervals, subgroup metrics, missingness stress tests and decision curves.

Bootstrap intervals condition on this split and fitted model; they do not account for model-search uncertainty or population shift. Subgroup scores are descriptive, potentially unstable, and do not establish fairness. Random missingness stress tests are not real hospital external validation. Decision curves describe hypothetical threshold trade-offs; benefits have not been confirmed in a clinical trial. No oversampling or synthetic patient records are used.

## API and deployment boundary

```powershell
# Set a unique secret locally; do not commit or share it.
$env:HEARTLENS_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
.\.venv\Scripts\python.exe -m uvicorn heart_research.risk_api:app --host 127.0.0.1 --port 8000
```

Send POST `/predict` with `X-API-Key` and JSON `{ "feature_set": "no_lab", "measurements": { ... } }`. Full field names are in `examples/risk_request.json`. `/health` exposes readiness only. Input fields are allowlisted, nonfinite values and implausible ranges are refused, required measurements cannot be missing, and unknown outcome columns are rejected. Optional missing fields are returned explicitly. Every response identifies the CHD endpoint and research-only status. No patient records are deliberately persisted by the app. Do not upload model pickle files; joblib can execute code. Load only administrator-trusted artifacts. Checksums detect file mismatch, not a malicious actor replacing both model and manifest.

This is a local research deployment foundation. Internet hosting would additionally need a TLS gateway, managed secrets, rate limits, authorization, audit retention and privacy controls, dependency vulnerability review, operational monitoring and load testing. Docker support is a packaging convenience, not proof these controls exist.

Optional container commands (the Docker engine was unavailable in the development environment, so this container has not been runtime-verified):

```powershell
docker build -f Dockerfile.risk -t heartlens-risk .
docker run --rm -p 127.0.0.1:8000:8000 --env HEARTLENS_API_KEY --mount "type=bind,source=$((Resolve-Path artifacts/affordable-risk-v2).Path),target=/models,readonly" heartlens-risk
```

Clinical release requires an adjudicated target endpoint (first MI, hard CHD or broader CVD), an appropriate prospective cohort with event times/censoring, approved access and use rights, patient/site/time separated validation, independent contemporary target-population testing, calibration assessment, a clinician-agreed operating policy, human-factors evaluation, monitoring and applicable regulatory review. These release gates remain unmet.

For a strict heart-attack-risk major project, replace the teaching CSV only after obtaining a suitable longitudinal MI cohort. Do not rename CHD, prevalent disease, post-MI complications or synthetic labels as future MI.

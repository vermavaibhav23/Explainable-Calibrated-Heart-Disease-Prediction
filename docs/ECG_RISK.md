# Routine checkup + ECG cardiovascular risk

This is the extension matching the revised scope: routine blood tests and resting ECG measurements are allowed, and a broader cardiovascular outcome is acceptable. The trained endpoint is **10-year death from diseases of the heart or cerebrovascular disease**. It is not first-heart-attack risk, nonfatal cardiovascular disease, or acute ECG diagnosis.

## Run locally in VS Code

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-risk.txt
.\.venv\Scripts\python.exe scripts/download_nhanes.py
.\.venv\Scripts\python.exe -m heart_research.nhanes
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m heart_research.ecg_risk --jobs 2 --iterations 6
.\.venv\Scripts\python.exe scripts/package_ecg.py
.\.venv\Scripts\python.exe -m streamlit run ecg_app.py
```

Official downloads are about 320 MB. Raw files, participant-level predictions and model artifacts stay local and are git-ignored. A new training run refuses to overwrite an existing directory. The two earlier project experiments remain separately reproducible.

## Data and affordable inputs

The [CDC NHANES III release](https://wwwn.cdc.gov/nchs/nhanes/nhanes3/datafiles.aspx) supplies household history, examinations, laboratories and resting ECG summaries. The [public 2019 mortality linkage](https://www.cdc.gov/nchs/linked-data/mortality-files/index.html) provides follow-up. CDC states that mortality status is unchanged but some public follow-up times and underlying causes are altered for confidentiality. The source is suitable for public-data research; those alterations remain an outcome-measurement limitation.

The preparation retains 8,486 participants aged 40–89 with eligible linkage, positive examination follow-up, an identifiable family, positive MEC weight and an ascertainable ten-year endpoint. There are 839 events. The ECG source contains 8,561 records; exclusion counts and all source checksums are recorded in the data audit. Ages 90+ are excluded because the public age field is top-coded. Prior disease is not excluded, so this is not a pure primary-prevention cohort.

The routine set uses age, sex as recorded, blood pressure, BMI, smoking, known diabetes, BP medication, previous MI, total/HDL cholesterol, HbA1c and standardized creatinine. The ECG set adds heart rate, PR/QRS/QT intervals, P/QRS/T axes, and a major-abnormality flag using the source's Minnesota criteria. QT means uncorrected QT, not QTc. An ECG specialist/software report must supply these measurements; this project does not extract them from photographs or waveform files.

No angiography, coronary calcium scan, stress imaging, troponin panel or genetic testing is needed. Test affordability still varies by setting; no local cost study was performed.

The fixed-width reader obtains selected column positions from CDC's SAS specifications without executing the SAS. Special missing-value codes are converted to missing values; skip-pattern information is used for non-smokers and people not prescribed BP medication. Missingness is otherwise retained for training-fold imputation. Historical creatinine is corrected as `0.960 * original - 0.184`, following the [CDC laboratory documentation, page 111](https://wwwn.cdc.gov/nchs/data/nhanes3/1a/lab-acc.pdf). Modern inference expects standardized creatinine directly.

## Target construction and validation

- Positive: eligible linked participant who died within 120 months after the examination with leading cause 001 (heart disease) or 005 (cerebrovascular disease).
- Negative: known alive for at least 120 months, death after that horizon, or known competing-cause death before the horizon. Treating an early competing death as a non-event estimates cumulative incidence of fatal CVD; it does not estimate survival conditional on avoiding other deaths.
- Excluded: unknown early death cause, insufficient surviving follow-up, ineligible linkage, zero/invalid examination follow-up or unsupported baseline metadata. No outcome is inferred from an ECG diagnosis.
- Development: survey phase 1 (1988–1991), split by family into training and calibration. Test: all eligible phase 2 participants (1991–1994). Family identifiers never enter the predictors and do not overlap across partitions.
- Five-fold stratified, family-grouped development search covers eight model families, six parameter configurations each, for each predeclared feature set. Pipelines fit imputation, indicators, scaling and optional splines inside folds. A top-three probability blend competes against the strongest individual development model.
- Phase-specific MEC weights enter model fitting, AUROC selection, calibration comparison and primary test metrics. Calibration chooses among none, sigmoid and isotonic using grouped calibration OOF Brier score. A weighted calibration sensitivity target of 80% determines the research screening threshold before test evaluation.
- The test phase is evaluated after model, calibrator and threshold are frozen. Report both weighted and raw sample results, a constant-probability baseline, subgroup results, ROC/precision-recall/calibration plots and hypothetical decision curves.
- Reported intervals use rescaled pseudo-PSU resampling within phase-specific strata. Each stratum here contains two PSUs; the bootstrap selects `n_h - 1` PSUs per stratum, with the common weight multiplier cancelling in normalized metrics. Intervals are conditional on the fitted model and omit finite-population corrections. They do not establish transportability or quantify training-search uncertainty.

Development OOF scores are used for selection and are optimistic. Temporal validation is stronger evidence than a random split alone, but both phases belong to the same historical US survey. It is not validation at an independent contemporary hospital or in India. Subgroup results do not certify fairness. The ECG-versus-routine comparison uses selected pipelines; any difference cannot be interpreted as the causal benefit of ordering an ECG.

## App, API and release status

### Later survey-cycle evaluation

After packaging the frozen models, run `python scripts/external_nhanes.py` to download official NHANES 2007–2008 demographic, examination, laboratory and questionnaire files and their 2019 public mortality linkage. It evaluates the routine-only model without refitting, recalibration or threshold changes. Ages are restricted to 40–79 because that release top-codes older ages. The released 2007 creatinine is already standardized; the NHANES III correction must not be applied again. Matching ECG summaries are unavailable, so this cannot externally validate the ECG model.

The completed check includes 3,473 participants and 153 events. Weighted AUROC is 0.813 (conditional 95% interval 0.765–0.864), sensitivity 68.6%, specificity 80.4%, and precision 10.5%. Mean predicted risk is 5.2% against observed 3.2%. Population, age, treatment and measurement differences matter: the observed overprediction and sensitivity loss prevent a claim of deployment readiness. Full weighted and unweighted results, missingness and source hashes are in `reports/ecg_risk/external_2007/results.json`.

Run `python scripts/export_reference.py` after both experiments and external evaluation to export aggregate reports. It excludes participant predictions, raw records and model binaries. The public protocols retain split hashes; the full local protocols retain split membership for audit.

`ecg_app.py` supports both input sets, final-model SHAP explanations and the temporal evaluation report. The original `risk_app.py` remains the separate Framingham CHD demonstration.

The authenticated API in `heart_research.risk_api` exposes `/predict-cvd-mortality` when the packaged ECG artifacts are installed. It expects `feature_set` (`routine` or `routine_ecg`) and numeric/null `measurements`. The API key requirement, local binding, strict schema and checksum checks described in AFFORDABLE_RISK.md apply. Use `HEARTLENS_ECG_ARTIFACTS` to change the artifact path. Missing ECG artifacts return 503, not an invented probability. The Docker configuration is prepared but was not runtime-tested because the local engine was unavailable.

`examples/ecg_request.json` contains a fictional request. For container use, mount the whole local `artifacts` directory read-only and set both `HEARTLENS_RISK_ARTIFACTS=/artifacts/affordable-risk-v2` and `HEARTLENS_ECG_ARTIFACTS=/artifacts/ecg-risk-v3` alongside `HEARTLENS_API_KEY`. No credential belongs in source control.

The model is a research deliverable, **not an industry-ready clinical system**. Independent contemporary validation, modern laboratory/ECG harmonization, target-country calibration, clinical utility testing, clinician-agreed use/thresholds, privacy/security review and operational monitoring are unresolved. The code can be demonstrated now; a medical deployment claim cannot be supported by this study alone.

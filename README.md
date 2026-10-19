# HeartLens

**Explainable and Calibrated Heart Disease Prediction System with Robust Validation**

A complete local Python / VS Code major project extending [EnsembleHeartDiseasePrediction](https://github.com/vermavaibhav23/EnsembleHeartDiseasePrediction). It includes a reproducible research pipeline, seven model/ablation comparisons, a calibrated voting ensemble, explanations of the actual deployed probability function, and an interactive Streamlit application. No Colab notebooks are required.

This is a research prototype for classifying the dataset's `HeartDisease` label. It does not diagnose disease, predict a future heart attack, or estimate a time-specific clinical risk.

## Executed results

The full five-fold run completed on 19 October 2026 using all 918 retained records. The calibrated ensemble achieved **87.15% accuracy, 90.35% sensitivity, 83.17% specificity, 0.929 ROC-AUC and 0.0986 Brier score** on pooled outer-fold predictions. Calibration slightly improved Brier score relative to soft voting but did not improve accuracy. These are internal-validation results, not clinical or external validation. See [results and interpretation](docs/RESULTS.md) and [aggregate run artifacts](reports/reference_run).

Verification: 14 unit/application-startup tests and 3 local full-artifact integration tests passed; GitHub Actions also passed its source-only checks. The CLI successfully generated SHAP explanations for all three fictional example records.

## Start in VS Code (Windows)

Requires Python 3.11–3.13 and VS Code with the Python extension.

```powershell
# Open the folder in VS Code, then use Terminal > New Terminal.
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

# Download the dataset from its original Kaggle source.
.\.venv\Scripts\python.exe -m heart_research.cli download

# Full experiment, including nested validation and final demonstration model.
.\.venv\Scripts\python.exe -m heart_research.cli train --jobs 2

# Start the app; visit http://localhost:8501.
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `data/raw/heart.csv` or `artifacts/full` already exists in the delivered local project, reuse it; no download or retraining is needed to start the app. To repeat an experiment, use a new output directory, for example `--output artifacts/run-02`. Set `$env:HEART_RUN_DIR='artifacts/run-02'` before launching the app to load that run. Run folders containing a manifest are never overwritten by training.

Select `.venv\Scripts\python.exe` using **Python: Select Interpreter**. Press **F5** and select **HeartLens: Streamlit app** to debug. On macOS/Linux, create a venv with `python -m venv .venv`, use `.venv/bin/python`, then run the same modules.

## What the project adds

| Area | Implementation |
|---|---|
| Baselines | Logistic regression, ExtraTrees, probabilistic SVM |
| Ensembles | Soft voting and sigmoid-calibrated soft voting |
| Ablations | Calibrated voting without interactions; reduced-input calibrated model |
| Validation | Five outer stratified folds, three-fold inner hyperparameter search, training-only threshold selection |
| Calibration | Three-fold sigmoid calibration inside each outer training set; Brier score, log loss, reliability plot |
| Metrics | Accuracy, balanced accuracy, sensitivity, specificity, precision, F1, ROC-AUC, average precision |
| Uncertainty | Conditional 95% stratified bootstrap intervals from outer out-of-fold predictions |
| Missing inputs | Imputation, missing indicators, named-field and random-missingness stress tests |
| Subgroups | Age-band and sex-specific metrics with sample counts |
| Explainability | Kernel SHAP for the exact final calibrated pipeline, grouped at original input level |
| Application | Individual prediction, individual SHAP chart, CSV prediction, evidence dashboard, downloadable HTML reports |
| Engineering | Package/CLI, isolated environment, tests, CI, VS Code launch settings and data/model provenance |

## Reproduce or extend the study

```powershell
# Fast software check, NOT the final research protocol.
.\.venv\Scripts\python.exe -m heart_research.cli train --smoke --output artifacts/smoke

# Batch prediction with explanations; examples are fictional.
.\.venv\Scripts\python.exe -m heart_research.cli predict --input examples/patients.csv --explain

# Optional external evaluation; exact overlap with training is rejected.
.\.venv\Scripts\python.exe -m heart_research.cli external --data path/to/independent.csv

# Verify the implementation.
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

An external CSV must use the same schema, units and outcome definition. A passed overlap check only excludes identical feature records; it does not prove patient independence. Do not use a subset of the same pooled source as an independent external test set. External validation is supported but is not claimed for the included run.

## Evaluation protocol

1. Validate schema, convert zero BP/cholesterol to missing, remove exact duplicate feature records, reject conflicting labels.
2. Split the retained records into stratified outer folds. Each record contributes one evaluation prediction per model.
3. Inside each outer training fold: tune a small pre-specified grid by Brier score using inner CV. Calibration is itself fitted using training-only splits. All imputers, encoders and scalers are inside the estimator.
4. Select a threshold using development out-of-fold predictions: maximize specificity subject to a 0.90 development sensitivity target. This is a study design choice, not a clinically approved operating point.
5. Evaluate once on the outer validation fold; retain its probabilities, threshold and metrics.
6. Pool outer predictions for summaries and bootstrap intervals. Evaluate missingness using those same fitted outer models and thresholds.
7. Fit the **pre-specified calibrated voting model** on all development data for the application. The dashboard reports nested-CV results, not final-model training accuracy. The deployed family is not chosen by ranking the outer results.

Hyperparameter tuning and threshold-development CV reuse training labels, so those internal threshold estimates can be optimistic. They are never presented as unbiased performance; the untouched outer folds evaluate the complete procedure. Dataset reuse from the minor project still limits independence of the research design. See [methodology](docs/METHODOLOGY.md).

## Files and outputs

```text
app.py                         Streamlit application
src/heart_research/             Data, models, evaluation, SHAP, reports and CLI
tests/                         Regression and application tests
.vscode/                       Interpreter and debugger configurations
scripts/                       Windows setup, run and GitHub publishing
examples/patients.csv           Fictional demonstration records
data/raw/                      Downloaded data (ignored by Git)
artifacts/full/                 Fitted model, metrics, charts, audit and HTML report
reports/reference_run/         Aggregate results of the executed full experiment
docs/                          Methodology, model card and major-project guide
```

Models and raw/background records stay local and are excluded from Git. Never load an untrusted `.joblib` file: deserialization can execute code. The public repository carries source and aggregate results; users reproduce the model locally. The local app does not persist uploaded records.

## Dataset and attribution

Original data: [Heart Failure Prediction Dataset by fedesoriano on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). The downloader writes a provenance file with URL and SHA-256. Upstream dataset terms apply; raw data are not redistributed in this repository. The original project is credited above. The input is tabular clinical/exercise-test data, not raw ECG waveforms. See [data dictionary](docs/DATA_DICTIONARY.md).

## Publish the new repository

```powershell
gh auth login
powershell -ExecutionPolicy Bypass -File scripts/publish.ps1
```

If `origin` is already configured, the script pushes `main` to that repository. Otherwise it creates a **public** repository named `Explainable-Calibrated-Heart-Disease-Prediction` in the signed-in account and pushes the existing local Git history. It does not modify the minor-project repository.


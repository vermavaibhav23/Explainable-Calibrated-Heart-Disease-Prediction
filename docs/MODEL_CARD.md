# Model card

**System:** HeartLens, version 1.0.0. **Task:** binary classification of the public dataset's HeartDisease label.

**Intended use:** student research, reproducibility, software demonstration and exploration of calibration, explanations and missing-input robustness. **Out of scope:** diagnosis, treatment recommendations, population screening deployment or future-event risk prediction.

**Architecture:** logistic regression, ExtraTrees and SVM soft voting; sigmoid calibration of the complete pipeline; development-selected operating threshold. The deployed family is pre-specified. Data cleaning and feature engineering are consistent between training and inference.

**Data:** Kaggle Heart Failure Prediction Dataset by fedesoriano. The run audit records row counts, missingness, duplicates, class counts and a file checksum. No independent external cohort is included. Source overlap and prior use in the minor project limit validation independence.

**Performance:** consult `reports/reference_run/model_comparison.csv` for the executed research run, `confidence_intervals.json` for conditional intervals and `fold_metrics.csv` for fold variation. Numerical performance is intentionally not hardcoded here. `manifest.json` records the protocol, dataset hash, library versions and deployed threshold. The Streamlit app reads results from its selected local artifact directory.

**Limitations:** retrospective small dataset, no patient/site IDs, potential dataset shift, approximate SHAP, simulated missingness, limited subgroup sizes and development-selected sensitivity target. Conditional bootstrap intervals do not capture full model retraining uncertainty.

**Input behavior:** missing inputs are imputed from training data; unknown categories and invalid values are rejected. The application flags values outside observed training ranges and probabilities within 0.05 of the operating threshold. This proximity flag is not an uncertainty estimate. No individual clinical confidence interval is claimed.

**Privacy and security:** local app by default, bound to 127.0.0.1. No external inference API. Uploaded batch CSVs are processed in memory. Model artifacts contain a small raw background sample and therefore stay local with the raw data. Only source and aggregate reference metrics are intended for publication. `.joblib` must come from a trusted local training run.


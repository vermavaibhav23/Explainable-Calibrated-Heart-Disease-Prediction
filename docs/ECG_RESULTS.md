# Routine checkups and ECG: completed temporal validation

**Endpoint: 10-year death from heart or cerebrovascular disease.** Not nonfatal CVD, first heart attack, or acute diagnosis.

Cohort: 8,486 NHANES III participants aged 40–89, with 839 events. Train/calibration/test counts: {'train': 3489, 'calibration': 871, 'test': 4126}. Earlier survey phase develops the model; later phase is reserved for testing. Family members stay together. MEC survey weights are used in fitting, selection, calibration and reported primary metrics.

| Inputs | Selected models | Weighted AUROC | Sensitivity | Specificity | Precision | Screening accuracy | Accuracy at 0.5 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| routine | catboost, extra_trees, random_forest | 0.871 | 81.5% | 79.0% | 21.6% | 79.2% | 93.1% | 0.0519 |
| routine_ecg | catboost, extra_trees, logistic | 0.886 | 85.4% | 76.7% | 20.7% | 77.3% | 93.3% | 0.0509 |

## routine

Weighted AUROC interval: 0.852–0.894; sensitivity interval: 76.6%–87.7%; specificity interval: 77.0%–80.9%.
Unweighted screening counts: TP 327, FN 61, FP 1180, TN 2558. These counts should not be used to recompute the weighted percentages.
Calibration method: none; research screening threshold: 0.076325. Constant baseline weighted accuracy: 93.4%, sensitivity zero, AUROC 0.5, Brier 0.0620.

## routine_ecg

Weighted AUROC interval: 0.866–0.906; sensitivity interval: 81.3%–90.8%; specificity interval: 74.7%–78.7%.
Unweighted screening counts: TP 342, FN 46, FP 1246, TN 2492. These counts should not be used to recompute the weighted percentages.
Calibration method: none; research screening threshold: 0.063418. Constant baseline weighted accuracy: 93.4%, sensitivity zero, AUROC 0.5, Brier 0.0620.

## Limits of the result

The confidence intervals condition on this fitted model; they use rescaled pseudo-PSU resampling within strata and omit finite-population corrections. They do not cover training/search uncertainty or modern population shift. Predictions use a historical cohort that includes pre-existing disease. Public mortality causes/times are modified for some records to protect confidentiality. No external contemporary or Indian validation was performed.

Accuracy at 0.5 is dominated by the much more common absence of a fatal event. The screening threshold intentionally makes a different sensitivity/specificity trade-off. Nonfatal events are outside this endpoint. The scores cannot be directly ranked against the minor project's existing-disease accuracy, the Framingham CHD benchmark, or unrelated IEEE diagnosis studies.

This is a trained research system, not an industry-ready clinical model. Read ECG_RISK.md for ingestion, endpoint, validation and release requirements.

## Paired ECG comparison and later-cohort check

ECG augmentation changed weighted AUROC by +0.0147, with conditional paired 95% interval +0.0035 to +0.0243. This compares full selected pipelines, not a causal effect of ordering tests.

The frozen routine-only model was also evaluated without refitting, recalibration or threshold changes on 3,473 NHANES 2007–2008 participants aged 40–79 (153 fatal events). Weighted AUROC 0.813; sensitivity 68.6%; specificity 80.4%; precision 10.5%; screening accuracy 80.0%; Brier 0.0317. Mean predicted risk 5.2% versus observed risk 3.2%. This is a later independent survey-cycle check of the routine model, not validation of the ECG model. Age restrictions, case mix and measurement differences prevent direct score ranking across cohorts.

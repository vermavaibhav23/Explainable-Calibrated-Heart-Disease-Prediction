# Executed reference experiment

Run date: **19 October 2026**. Dataset: **918 retained records**, 508 positive and 410 negative labels. Five stratified outer folds, three-fold inner search, three-fold calibration, seed 42. Full protocol, not smoke mode. Every record has one outer-fold prediction per candidate.

## Model comparison

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 87.58% | 91.14% | 83.17% | 0.9283 | 0.1013 |
| ExtraTrees | 86.82% | 89.57% | 83.41% | 0.9303 | 0.1000 |
| SVM | 86.60% | 90.94% | 81.22% | 0.9206 | 0.1045 |
| Soft voting | 87.47% | 91.14% | 82.93% | 0.9307 | 0.0997 |
| Calibrated voting | 87.15% | 90.35% | 83.17% | 0.9292 | 0.0986 |
| Calibrated, no interactions | 86.93% | 90.16% | 82.93% | 0.9275 | 0.0991 |
| Calibrated, reduced inputs | 78.98% | 89.76% | 65.61% | 0.8595 | 0.1512 |

Calibrated voting was the pre-specified deployment family. Its Brier score was slightly lower than uncalibrated voting, while accuracy and ROC-AUC were slightly lower. Logistic regression had the highest observed accuracy. There is no basis here for claiming statistically significant superiority of the ensemble. These model comparisons are part of the study, not a reason to silently replace the pre-specified deployed model after examining validation scores.

The no-interaction and reduced-input variants use equal voting weights; the full model searches equal versus ExtraTrees-emphasized weights. Thus these are configuration comparisons, not a perfectly isolated causal estimate of feature engineering. A future controlled ablation should hold the weighting/search strategy identical. The observed interaction difference is small and should not be overstated.

## Calibrated model uncertainty

Conditional 95% stratified bootstrap intervals (1,000 resamples):

| Metric | Estimate | Interval |
|---|---:|---:|
| Accuracy | 87.15% | 85.08–89.11% |
| Sensitivity | 90.35% | 87.80–92.72% |
| Specificity | 83.17% | 79.51–86.83% |
| ROC-AUC | 0.9292 | 0.9123–0.9450 |
| Brier score | 0.0986 | 0.0853–0.1130 |

Intervals condition on the saved outer predictions. They do not include full model-retraining variability or uncertainty from choosing this research design after the minor project used the same data.

## Missing-input findings

Masking all cholesterol values reduced accuracy to **78.10%**. Masking all ST-slope values reduced it to **76.47%**. Independently masking 10% and 30% of cells reduced accuracy to **83.88%** and **73.09%**, respectively. Sensitivity sometimes rose while specificity deteriorated: missingness shifted predictions toward the positive class. This is a useful failure-mode finding rather than evidence that the system is robust to arbitrary missing inputs.

## Final application model

The final all-data model uses voting weights `[1, 2, 1]` for logistic regression, ExtraTrees and SVM, and a development-selected threshold of **0.54**. Each outer fold had its own training-selected threshold; do not apply 0.54 retroactively to the outer evaluation table. The application model's training accuracy is not reported as validation performance.

All numerical source tables, fold settings, dataset audit and hashes are in `reports/reference_run/`. The local `artifacts/full/report.html` contains the tables, intervals and plots in one self-contained report. Raw records, per-record validation predictions and fitted model/background data are excluded from the public repository.


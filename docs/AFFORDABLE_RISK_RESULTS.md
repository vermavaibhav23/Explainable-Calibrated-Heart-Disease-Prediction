# Affordable CHD risk: completed reference experiment

Educational research result; not MI-only prediction or an externally validated clinical model.

Models were selected on development data and calibrated on a separate partition before evaluating 848 held-out records.

| Inputs | Selected model(s) | Test AUROC | Sensitivity | Specificity | Precision | Screening accuracy | Accuracy at 0.5 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no_lab | logistic, spline_logistic, catboost | 0.718 | 77.5% | 53.3% | 22.9% | 57.0% | 85.1% | 0.1179 |
| basic_lab | logistic, spline_logistic, catboost | 0.717 | 80.6% | 51.5% | 23.0% | 55.9% | 85.4% | 0.1175 |

A constant no-event prediction achieves about 84.8% accuracy while detecting no events. Screening accuracy uses the calibration-selected sensitivity threshold and answers a different operating-policy question from accuracy at 0.5.

## Uncertainty and confusion counts

### no_lab

AUROC 95% conditional bootstrap interval: 0.672–0.764.
Sensitivity interval: 70.5%–84.5%; specificity: 49.8%–56.9%.
True positives 100, false negatives 29, false positives 336, true negatives 383.
Calibrator: none; research screening threshold: 0.123771.

| Subgroup | Records | Events | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| male=female | 485 | 63 | 65.1% | 63.3% |
| male=male | 363 | 66 | 89.4% | 39.1% |
| age=(0, 45] | 322 | 24 | 33.3% | 86.2% |
| age=(45, 60] | 406 | 71 | 81.7% | 37.3% |
| age=(60, 120] | 120 | 34 | 100.0% | 1.2% |

Subgroups are descriptive, not a fairness certification; small event counts make estimates uncertain.

### basic_lab

AUROC 95% conditional bootstrap interval: 0.672–0.763.
Sensitivity interval: 74.4%–87.6%; specificity: 48.1%–55.1%.
True positives 104, false negatives 25, false positives 349, true negatives 370.
Calibrator: none; research screening threshold: 0.116068.

| Subgroup | Records | Events | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| male=female | 485 | 63 | 69.8% | 61.6% |
| male=male | 363 | 66 | 90.9% | 37.0% |
| age=(0, 45] | 322 | 24 | 37.5% | 85.6% |
| age=(45, 60] | 406 | 71 | 85.9% | 34.3% |
| age=(60, 120] | 120 | 34 | 100.0% | 0.0% |

Subgroups are descriptive, not a fairness certification; small event counts make estimates uncertain.

## Interpretation

Adding basic labs changed AUROC by -0.0010 (paired conditional 95% interval -0.0096 to +0.0078). This compares the complete selected pipelines, not a pure causal effect of buying laboratory tests.

These models cannot be ranked against the old 918-row disease-detection project or unrelated IEEE diagnosis papers by accuracy. The target, prevalence, inputs and evaluation differ. This run does not show state-of-the-art or clinical readiness. See AFFORDABLE_RISK.md and IEEE_2026_COMPARISON.md for methodology, provenance and release gates.

The first run stopped during CatBoost development scoring due to a sklearn interface mismatch. An estimator adapter fixed this before any holdout evaluation; the original seed, splits and search space were retained.

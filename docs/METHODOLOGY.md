# Research methodology

## Question and hypotheses

How do probability calibration and missing clinical inputs affect ensemble classification reliability? Compare calibrated voting against uncalibrated voting and individual classifiers; test whether interactions help; quantify the effect of withholding fields. Improvement is a hypothesis, not a promised outcome.

The dataset contains a binary current heart-disease label, not a follow-up interval or future event label. Keep the task and paper title consistent with that outcome.

## Preprocessing

The schema contains 11 raw predictors. Zero cholesterol and resting blood pressure are interpreted as missing rather than as physiological measurements. This is a documented preprocessing assumption; consult source documentation before applying it to a different cohort. Numeric values use fold-specific median imputation, missing indicators and standardization. Categoricals use fold-specific mode imputation and one-hot encoding. Missing indicators created by `SimpleImputer` only cover columns missing during its fit. The pipeline gracefully imputes a newly missing field, but no newly fitted indicator is added at inference.

Four interactions are computed inside the pipeline: age×cholesterol, age×blood pressure, cholesterol×fasting-BS flag, and oldpeak×maximum heart rate. They are explicitly included in the numeric transformer. Their benefit is tested against a no-interactions ablation. They are exploratory mathematical interactions, not established clinical formulas.

The reduced-input model uses age, sex, chest pain, resting BP, cholesterol and fasting-BS flag. It removes exercise/ECG fields but still requires laboratory measurements; do not call it a no-test screening model.

## Models and selection

Fixed candidate families: logistic regression, ExtraTrees, SVM, soft vote, calibrated vote, no-interaction calibrated vote and reduced-input calibrated vote. The voting members are logistic regression, ExtraTrees and probabilistic SVM. SVM's internal probability fitting remains within the supplied training data. Sigmoid calibration wraps the entire preprocessing/voting pipeline. `ensemble=False` obtains cross-validated calibration predictions and refits the base pipeline on that training partition.

Hyperparameters are chosen by negative Brier score inside each outer training fold. Outer folds evaluate the entire selection/calibration/threshold procedure. Outer predictions do not select the deployed model family. No exact unbiased estimate is claimed for the internal CV score after tuning. For a larger follow-up study, add a separate threshold-selection layer or prospective validation cohort.

## Operating threshold

The default development target is sensitivity ≥0.90. Among thresholds meeting it, maximize specificity, breaking ties toward the higher threshold. Evaluate achieved sensitivity and specificity on outer validation records. The target is not a guaranteed held-out sensitivity. Calibration and classification thresholds answer different questions: probabilities versus actions. No clinical action is prescribed.

## Statistics and robustness

Report all folds and pooled OOF metrics, with stratified percentile bootstrap intervals. These are conditional intervals for the observed predictions and fixed empirical class mixture, not full repeated-training uncertainty. They do not establish statistical superiority. Models evaluated on the same records should be compared with paired methods in a future confirmatory analysis.

Stress tests mask cholesterol, BP or ST slope entirely, and independently mask 10% or 30% of input cells. The same fold-specific model and threshold predict the masked records. This simulates missing completely at random for the random conditions; it does not model clinically informative missingness. Subgroup metrics include denominators and undefined values where classes are absent. Small sample estimates are descriptive.

## Explanations

Kernel SHAP evaluates the final calibrated `predict_proba` function in probability space using 30 training background records. Categorical codes are reversible interface adapters; they are never fed as numeric categories to the original classifier. Masking occurs at the raw-feature level and interactions are recomputed, preserving the deployed feature path. Additivity is checked numerically. Attributions are approximate and depend on the background distribution. Correlated features and interventional masking can create implausible combinations; SHAP values are not causal effects or treatment advice.

## Limitations and future work

This is internal validation of a small, retrospective pooled dataset already used in the minor project. There is no untouched external cohort or prospective study. Exact duplicate removal cannot establish patient independence because patient/site identifiers are unavailable. Age and sex summaries do not prove fairness. Calibration may change under different prevalence, referral practices or measurement processes. Required exercise-test values limit the intended data collection setting.

Future work should prioritize a compatible, genuinely independent cohort with patient/site identifiers, clinically reviewed outcomes, and a registered evaluation protocol. Raw ECG analysis or multimodal fusion would require matched patient records and a separate task design.

## References

- [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn: nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [SHAP KernelExplainer](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html)
- [Original dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)


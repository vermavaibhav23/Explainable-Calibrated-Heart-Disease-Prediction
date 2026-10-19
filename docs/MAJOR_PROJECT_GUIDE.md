# Major-project guide

## Proposed title

Explainable and Calibrated Heart Disease Prediction System with Robust Validation

## Objectives

1. Correct the minor project's preprocessing and threshold-validation weaknesses.
2. Compare individual classifiers, voting and calibrated voting under a nested protocol.
3. Measure discrimination, sensitivity/specificity tradeoffs and probability reliability.
4. Evaluate missing-input robustness and age/sex subgroups.
5. Explain predictions of the actual deployed model and provide an interactive local application.

## Contribution and scope

This is an engineering and experimental extension, not a claim of a newly invented algorithm. The contribution is a reproducible evaluation of calibration and input availability, plus an integrated explanation/evidence application. Explainability alone and combining standard classifiers do not establish research novelty. A paper would need a careful related-work comparison and ideally a genuinely independent cohort.

## Suggested report chapters

1. Introduction, problem definition, scope and research questions.
2. Related work on tabular heart disease classification, calibration and explanations.
3. Dataset provenance, data dictionary and audit.
4. Architecture, preprocessing, candidate models and evaluation protocol.
5. Experimental results: all baselines, ablations, calibration, intervals, robustness and subgroups.
6. Application implementation, tests and demonstration.
7. Limitations, ethical considerations and future work.

Use actual generated tables and figures. Include findings where calibration or an ensemble does not improve performance. Do not promise 95% or 99% accuracy or relabel development accuracy as a test result.

## Demonstration sequence

Open the project in VS Code. Run the tests and show the CLI command. Open the Streamlit app, enter a fictional record, inspect the model probability and SHAP contributions, and download the HTML report. Open the evidence page and explain why Brier score differs from accuracy. Show missing-input degradation and subgroup counts. Upload the fictional batch template and export predictions. Close by identifying what an independent validation study would still need.

## Viva questions to prepare

- Why does tuning a threshold on the test labels bias the reported score?
- What is protected by the outer CV loop, and what happens inside the inner loop?
- Why can a calibrated model have lower accuracy but better probability reliability?
- How do imputation, interaction features and explanations stay consistent at inference?
- Why do SHAP contributions not prove causation?
- Why does a passed exact-duplicate check not prove patient independence?
- Why is this model not a future heart-attack risk predictor?


# 2026 IEEE literature comparison

Search checked 19 October 2026. This is a targeted review, not an exhaustive systematic review. Reported results are the authors' claims; no paper was independently reproduced. Cross-paper accuracy differences do not establish superiority when endpoints, populations, inputs, splits and prevalence differ.

| Paper / venue | Reported result | Task and inputs | Relationship to this project |
|---|---|---|---|
| Zhang et al., **2 Parallel Heart**, IEEE Transactions on Biomedical Engineering, early access 21 May 2026, DOI 10.1109/TBME.2026.3695354 | Integrated accuracy 96.6%; ECG AUROC 0.933; biochemical AUROC 0.960 | Acute MI identification combining ECG and 13 biochemical markers, including troponin and CK-MB | A journal comparator for multimodal diagnostics; not a benchmark for future risk without expensive tests. Abstract reviewed. |
| Goli and Sah, **Heterogeneous Committee-Based Adaptive Active Learning for Efficient Heart Disease Prediction**, IEEE Access, 2026, DOI 10.1109/ACCESS.2026.3659186 | Accuracy 96.04%, AUROC 0.9708, using 31.9% labelled data | XGBoost/Random Forest/logistic committee, adaptive querying and clustering; UCI-style disease diagnosis features | Annotation efficiency is a useful research direction, but the diagnostic endpoint and input set differ from longitudinal CHD. Author manuscript reviewed; exact split and preprocessing were not independently reproduced. |
| Rahmani et al., **DP-FedHybrid**, IEEE Access 14:64681–64695, 23 April 2026, DOI 10.1109/ACCESS.2026.3687059 | Accuracy 95.12% under standard federated learning; 94.15% under differential privacy | Federated CatBoost/Transformer stacking, privacy training | Useful privacy architecture reference. Abstract alone does not establish comparable cohort, endpoint or evaluation. Full paper verification remains necessary before replication. |

Primary sources: [2 Parallel Heart abstract](https://pubmed.ncbi.nlm.nih.gov/42166262/), [HC-AAL author manuscript](https://www.researchgate.net/publication/400249225_Heterogeneous_Committee-Based_Adaptive_Active_Learning_for_Efficient_Heart_Disease_Prediction), [DP-FedHybrid author institution record](https://research-portal.uws.ac.uk/en/publications/dp-fedhybrid-a-differentially-private-federated-stacking-framewor/).

A closer endpoint reference is **Integrating Artificial Intelligence into Healthcare: Predictive Analytics for Early and Equitable Diagnosis**, DOI 10.1109/ICAISET66439.2026.11542039, which reports Framingham CHD AUROC 0.698. It is an **IEEE 2026 conference paper, not a journal article**. Its abstract discusses calibration and sensitivity, but its split and preprocessing have not been reproduced here. Our score must not be described as a significant improvement over it without matched evaluation. [IEEE abstract](https://ieeexplore.ieee.org/abstract/document/11542039).

## What the major project can contribute

- Compare no-lab and basic-lab feature sets under an identical protected test split.
- Measure calibration, sensitivity/false-positive trade-offs and hypothetical decision benefit, rather than optimizing accuracy alone.
- Release the split protocol, full hyperparameter-search results, package versions, aggregate test results and limitations.
- Provide inference validation, artifact integrity checks, explicit missingness and endpoint-aware explanations.
- Add a truly independent cohort when access is approved. This is the strongest unresolved step toward a publishable and clinically useful result.

The present educational benchmark does not justify “best model,” state-of-the-art, prospective clinical reliability or industry-ready medical-device claims.

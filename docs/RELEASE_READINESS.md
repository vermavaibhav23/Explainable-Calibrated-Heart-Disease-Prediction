# Release target: 93% accuracy and 90% sensitivity

The user specified both targets on 19 October 2026, after the existing results were visible. Neither current model satisfies them. No model is authorized for clinical release. This document distinguishes executable software controls from evidence of medical performance.

## Executed feasibility audit

`python -m heart_research.release_check` reads the frozen predictions and metrics and writes `reports/ecg_risk/release_check.json`. It exits nonzero if the numerical targets fail. It does not retrain, change thresholds, replace artifacts or certify clinical readiness. The primary metrics are survey-weighted; unweighted diagnostics are also retained.

For the ECG model, a post-hoc search over every threshold reaches at most **71.0% weighted accuracy while retaining at least 90% sensitivity**. Requiring at least 93% accuracy permits at most **16.7% sensitivity**. These are optimistic diagnostic envelopes using known test labels, not deployable operating points or new validation. The current frozen threshold remains unchanged.

At this cohort's weighted event prevalence, 93% accuracy and exactly 90% sensitivity would require approximately **93.2% specificity**. Merely lowering the decision threshold, resampling the test set, or reporting the majority-class accuracy cannot meet the intended clinical objective. The routine-only later-cohort check also fails the joint gate.

## Implemented software controls

- Artifact checksum/schema verification before use; invalid or absent enabled bundles prevent readiness.
- Authenticated inference; credentials checked before JSON parsing.
- A 16 KiB request-body limit, including chunked requests, and a four-request in-flight bound per process. The server returns 503 on overload rather than accumulating an unbounded inference queue.
- Strict field/type/range validation. Schema-error responses do not repeat submitted medical inputs. No-store headers and server-generated request IDs accompany responses.
- Independent CVD-only deployment (`HEARTLENS_MODELS=cvd`), separate readiness endpoint, non-root container and container health check.
- Unit, service, trained-model and application tests. Local Docker runtime verification remains unavailable because its engine is not running.

Example local CVD service, after configuring a random secret through the environment:

```powershell
$env:HEARTLENS_MODELS='cvd'
$env:HEARTLENS_ECG_ARTIFACTS='artifacts/ecg-risk-v3'
python -m uvicorn heart_research.risk_api:app --host 127.0.0.1 --port 8000 --no-access-log --timeout-keep-alive 5
```

Before internet deployment, the operator must supply TLS termination, authenticated access management/key rotation, connection/body-read timeouts, rate limits, monitoring, incident response and validated backup/rollback procedures. The in-process bound is not a distributed rate limiter. Keep model directories read-only and trusted: a checksum beside a pickle is not a signature or protection against an attacker replacing both. Do not log patient bodies or API secrets at a proxy. None of these controls establishes medical fitness.

## Evidence still needed

A new model attempt needs a fixed intended population/outcome, representative longitudinal data with relevant predictors and reliable outcome linkage, and a fresh untouched validation cohort. The already inspected historical and 2007–2008 cohorts cannot become fresh tests by changing a random seed. Additional informative measurements, waveform ECG or longitudinal trends may help, but their benefit must be demonstrated rather than promised.

Select candidates using development-only validation with sensitivity constrained to at least 90%; select calibration/thresholds on separate calibration data. Freeze the complete candidate before new evaluation. Report both target metrics, precision, calibration, subgroup results and uncertainty on that new cohort. Prospective workflow evaluation and appropriate clinical/regulatory review remain necessary for a medical deployment claim. See [FDA/IMDRF good machine learning practice](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles) for representative data, independent testing and clinically relevant evaluation principles.

**Status: software hardened; clinical release blocked by unmet performance and validation requirements. No 93%/90% achievement is claimed.**

"""Audit joint accuracy/sensitivity requirements without changing model thresholds.

The threshold envelope uses known test labels and is explicitly an optimistic,
post-hoc diagnostic. It is never deployment evidence or threshold selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def threshold_envelope(y, probability, weight, accuracy=0.93, sensitivity=0.90):
    y, probability, weight = map(np.asarray, (y, probability, weight))
    if not (len(y) == len(probability) == len(weight)) or len(y) == 0:
        raise ValueError("Aligned nonempty arrays are required.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary outcome classes are required.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Weights must be finite and positive.")
    fpr, tpr, _ = roc_curve(y, probability, sample_weight=weight, drop_intermediate=False)
    prevalence = np.average(y, weights=weight)
    acc = prevalence * tpr + (1 - prevalence) * (1 - fpr)
    return {
        "post_hoc_diagnostic_only": True,
        "joint_target_achievable_by_any_threshold_on_this_sample": bool(((acc >= accuracy) & (tpr >= sensitivity)).any()),
        "maximum_accuracy_at_required_sensitivity": float(acc[tpr >= sensitivity].max()),
        "maximum_sensitivity_at_required_accuracy": float(tpr[acc >= accuracy].max()) if (acc >= accuracy).any() else None,
        "prevalence": float(prevalence),
        "specificity_needed_at_exact_target_sensitivity": float((accuracy - prevalence * sensitivity) / (1 - prevalence)),
    }


def acceptance(metrics, accuracy=0.93, sensitivity=0.90):
    """A numerical gate only; passing cannot certify clinical readiness."""
    values = [metrics.get("accuracy"), metrics.get("sensitivity")]
    valid = all(isinstance(v, (float, int)) and not isinstance(v, bool) and np.isfinite(v) and 0 <= v <= 1 for v in values)
    return bool(valid and values[0] >= accuracy and values[1] >= sensitivity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/ecg-risk-v3"))
    parser.add_argument("--output", type=Path, default=Path("reports/ecg_risk/release_check.json"))
    args = parser.parse_args()
    report = {"required_accuracy": 0.93, "required_sensitivity": 0.90,
              "clinical_release_authorized": False,
              "scope": "Existing-model audit; targets specified after results were seen. Fresh validation required for any new candidate.",
              "models": {}}
    for name in ["routine", "routine_ecg"]:
        rows = pd.read_csv(args.run / name / "test_predictions.csv")
        result = json.loads((args.run / name / "test_metrics.json").read_text())
        report["models"][name] = {
            "frozen_threshold": result["threshold"],
            "weighted_frozen_metrics": result["weighted_screening"],
            "frozen_numerical_gate_passed": acceptance(result["weighted_screening"]),
            "weighted_threshold_envelope": threshold_envelope(rows.target, rows.probability, rows.weight),
            "unweighted_threshold_envelope": threshold_envelope(rows.target, rows.probability, np.ones(len(rows))),
        }
    external = args.run / "external_2007" / "results.json"
    if external.exists():
        report["external_routine_numerical_gate_passed"] = acceptance(json.loads(external.read_text())["weighted_screening"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    # Deliberately nonzero when a current candidate fails the user's target.
    return 0 if all(m["frozen_numerical_gate_passed"] for m in report["models"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

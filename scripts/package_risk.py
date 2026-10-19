"""Audit, checksum and render a completed risk run; never refit or retune it."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from heart_research.risk import FEATURE_SETS, load_risk_data, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    protocol = json.loads((root / "protocol.json").read_text())
    if hashlib.sha256(Path(args.data).read_bytes()).hexdigest() != protocol["sha256"]:
        raise ValueError("Dataset differs from frozen run.")
    data = load_risk_data(args.data)
    partitions = protocol["splits"]
    sets = {name: set(rows) for name, rows in partitions.items()}
    assert all(not sets[a] & sets[b] for a, b in [("train", "calibration"), ("train", "test"), ("calibration", "test")])
    assert set.union(*sets.values()) == set(range(len(data)))
    audit = {"partition_rows_disjoint": True, "feature_profiles_disjoint": {}}
    for name, features in FEATURE_SETS.items():
        hashes = pd.util.hash_pandas_object(data[features], index=False)
        identities = {k: set(hashes.iloc[v]) for k, v in partitions.items()}
        disjoint = all(not identities[a] & identities[b] for a, b in
                       [("train", "calibration"), ("train", "test"), ("calibration", "test")])
        audit["feature_profiles_disjoint"][name] = disjoint
        if not disjoint:
            raise ValueError(f"{name} baseline profiles overlap; do not publish this run as independent.")
        data.iloc[partitions["train"]][features].sample(30, random_state=2026).to_csv(
            root / name / "explanation_background.csv", index=False)
    manifest = {"version": "2.0.0-research", "endpoint": protocol["endpoint"], "research_only": True,
                "clinically_validated": False, "data_sha256": protocol["sha256"], "model_sha256": {},
                "versions": protocol["versions"]}
    summary = json.loads((root / "summary.json").read_text())
    lines = ["# Affordable CHD risk: completed reference experiment", "",
             "Educational research result; not MI-only prediction or an externally validated clinical model.", "",
             "Models were selected on development data and calibrated on a separate partition before evaluating 848 held-out records.", "",
             "| Inputs | Selected model(s) | Test AUROC | Sensitivity | Specificity | Precision | Screening accuracy | Accuracy at 0.5 | Brier |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in summary.items():
        folder = root / name
        manifest["model_sha256"][name] = hashlib.sha256((folder / "model.joblib").read_bytes()).hexdigest()
        m = result["screening"]
        lines.append(f"| {name} | {', '.join(result['selected_models'])} | {m['roc_auc']:.3f} | "
                     f"{m['sensitivity']:.1%} | {m['specificity']:.1%} | {m['precision']:.1%} | "
                     f"{m['accuracy']:.1%} | {result['threshold_0_5']['accuracy']:.1%} | {m['brier']:.4f} |")
        predictions = pd.read_csv(folder / "holdout_predictions.csv")
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        RocCurveDisplay.from_predictions(predictions.y, predictions.probability, ax=axes[0, 0])
        axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
        PrecisionRecallDisplay.from_predictions(predictions.y, predictions.probability, ax=axes[0, 1])
        axes[0, 1].axhline(predictions.y.mean(), linestyle="--", color="gray")
        actual, predicted = calibration_curve(predictions.y, predictions.probability, n_bins=8, strategy="quantile")
        axes[1, 0].plot(predicted, actual, "o-")
        axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
        axes[1, 0].set(xlabel="Mean predicted risk", ylabel="Observed fraction", title="Calibration (quantile bins)")
        curve = pd.read_csv(folder / "decision_curve.csv")
        for key in ["model_net_benefit", "all_net_benefit", "none_net_benefit"]:
            axes[1, 1].plot(curve.threshold, curve[key], label=key)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set(xlabel="Hypothetical decision threshold", ylabel="Net benefit", title="Decision curve")
        fig.suptitle(f"HeartLens {name}: untouched test, n={len(predictions)}")
        fig.savefig(folder / "evaluation.png", dpi=150)
        plt.close(fig)
    lines += ["", "A constant no-event prediction achieves about 84.8% accuracy while detecting no events. Screening accuracy uses the calibration-selected sensitivity threshold and answers a different operating-policy question from accuracy at 0.5.", "",
              "## Uncertainty and confusion counts", ""]
    for name, result in summary.items():
        m = result["screening"]
        ci = result["ci95_screening"]
        lines += [f"### {name}", "", f"AUROC 95% conditional bootstrap interval: {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f}.",
                  f"Sensitivity interval: {ci['sensitivity'][0]:.1%}–{ci['sensitivity'][1]:.1%}; specificity: {ci['specificity'][0]:.1%}–{ci['specificity'][1]:.1%}.",
                  f"True positives {m['tp']}, false negatives {m['fn']}, false positives {m['fp']}, true negatives {m['tn']}.",
                  f"Calibrator: {result['calibration']}; research screening threshold: {result['threshold']:.6f}.", ""]
        groups = json.loads((root / name / "subgroups.json").read_text())
        lines += ["| Subgroup | Records | Events | Sensitivity | Specificity |", "|---|---:|---:|---:|---:|"]
        for group in groups:
            lines.append(f"| {group['attribute']}={group['group']} | {group['n']} | {group['positive_n']} | "
                         f"{group['sensitivity']:.1%} | {group['specificity']:.1%} |")
        lines += ["", "Subgroups are descriptive, not a fairness certification; small event counts make estimates uncertain.", ""]
    lines += ["## Interpretation", "",
              "These models cannot be ranked against the old 918-row disease-detection project or unrelated IEEE diagnosis papers by accuracy. The target, prevalence, inputs and evaluation differ. This run does not show state-of-the-art or clinical readiness. See AFFORDABLE_RISK.md and IEEE_2026_COMPARISON.md for methodology, provenance and release gates.", "",
              "The first run stopped during CatBoost development scoring due to a sklearn interface mismatch. An estimator adapter fixed this before any holdout evaluation; the original seed, splits and search space were retained."]
    save_json(root / "split_audit.json", audit)
    save_json(root / "bundle_manifest.json", manifest)
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Packaged and audited", root)


if __name__ == "__main__":
    main()

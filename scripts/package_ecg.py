"""Audit and report completed temporal ECG experiment; no model selection or refitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from heart_research.ecg_risk import design_bootstrap, weighted_metrics
from heart_research.nhanes import ENDPOINT
from heart_research.risk import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="artifacts/ecg-risk-v3")
    parser.add_argument("--data", default="data/raw/nhanes3_risk_standardized.csv")
    args = parser.parse_args()
    root = Path(args.run)
    data = pd.read_csv(args.data)
    protocol = json.loads((root / "protocol.json").read_text())
    assert hashlib.sha256(Path(args.data).read_bytes()).hexdigest() == protocol["sha256"]
    splits = {key: data.loc[data.id.isin(ids)] for key, ids in protocol["partitions"].items()}
    assert all(len(splits[k]) == len(protocol["partitions"][k]) for k in splits)
    for a, b in [("train_ids", "calibration_ids"), ("train_ids", "test_ids"), ("calibration_ids", "test_ids")]:
        assert not set(splits[a].family) & set(splits[b].family)
    assert splits["test_ids"].phase.eq(2).all()
    assert splits["train_ids"].phase.eq(1).all() and splits["calibration_ids"].phase.eq(1).all()
    summary = json.loads((root / "summary.json").read_text())
    manifest = {"version": "3.0.0-research", "endpoint": ENDPOINT, "clinically_validated": False,
                "model_sha256": {}, "data_sha256": protocol["sha256"], "split_audit": "family-disjoint temporal holdout",
                "interval_method": "Rao-Wu-style n_h-1 pseudo-PSU resampling within phase-specific two-PSU strata; conditional on fit; no FPC"}
    lines = ["# Routine checkups and ECG: completed temporal validation", "",
             "**Endpoint: 10-year death from heart or cerebrovascular disease.** Not nonfatal CVD, first heart attack, or acute diagnosis.", "",
             f"Cohort: {len(data):,} NHANES III participants aged 40–89, with {int(data.target.sum()):,} events. Train/calibration/test counts: {protocol['counts']}. Earlier survey phase develops the model; later phase is reserved for testing. Family members stay together. MEC survey weights are used in fitting, selection, calibration and reported primary metrics.", "",
             "| Inputs | Selected models | Weighted AUROC | Sensitivity | Specificity | Precision | Screening accuracy | Accuracy at 0.5 | Brier |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in summary.items():
        folder = root / name
        rows = pd.read_csv(folder / "test_predictions.csv")
        assert set(rows.id) == set(protocol["partitions"]["test_ids"]) and rows.id.is_unique
        # Recompute intervals at packaging with the explicit rescaled survey design.
        result["ci95_design_bootstrap"] = design_bootstrap(rows, rows.probability.to_numpy(), result["threshold"])
        result["interval_method"] = manifest["interval_method"]
        save_json(folder / "test_metrics.json", result)
        m = result["weighted_screening"]
        lines.append(f"| {name} | {', '.join(result['models'])} | {m['roc_auc']:.3f} | {m['sensitivity']:.1%} | "
                     f"{m['specificity']:.1%} | {m['precision']:.1%} | {m['accuracy']:.1%} | "
                     f"{result['weighted_at_0_5']['accuracy']:.1%} | {m['brier']:.4f} |")
        manifest["model_sha256"][name] = hashlib.sha256((folder / "model.joblib").read_bytes()).hexdigest()
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
        RocCurveDisplay.from_predictions(rows.target, rows.probability, sample_weight=rows.weight, ax=axes[0, 0])
        axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
        PrecisionRecallDisplay.from_predictions(rows.target, rows.probability, sample_weight=rows.weight, ax=axes[0, 1])
        axes[0, 1].axhline(np.average(rows.target, weights=rows.weight), linestyle="--", color="gray")
        bins = pd.qcut(rows.probability, 8, duplicates="drop")
        calibration = [(np.average(g.probability, weights=g.weight), np.average(g.target, weights=g.weight))
                       for _, g in rows.groupby(bins, observed=True)]
        observed = np.array(calibration)
        axes[1, 0].plot(observed[:, 0], observed[:, 1], "o-")
        axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
        axes[1, 0].set(xlabel="Weighted predicted risk", ylabel="Weighted observed fraction", title="Calibration")
        curves = []
        prevalence = np.average(rows.target, weights=rows.weight)
        for t in np.linspace(0.02, 0.25, 30):
            positive = rows.probability.ge(t)
            tp = rows.loc[positive & rows.target.eq(1), "weight"].sum()
            fp = rows.loc[positive & rows.target.eq(0), "weight"].sum()
            curves.append([t, (tp-fp*t/(1-t))/rows.weight.sum(), prevalence-(1-prevalence)*t/(1-t)])
        curve = pd.DataFrame(curves, columns=["threshold", "model", "all"])
        curve.to_csv(folder / "decision_curve.csv", index=False)
        axes[1, 1].plot(curve.threshold, curve.model, label="Model")
        axes[1, 1].plot(curve.threshold, curve["all"], label="All")
        axes[1, 1].axhline(0, color="gray", linestyle="--", label="None")
        axes[1, 1].legend()
        axes[1, 1].set(xlabel="Hypothetical decision threshold", ylabel="Weighted net benefit", title="Decision curve")
        fig.suptitle(f"{name}: later-phase test, n={len(rows)}, survey-weighted")
        fig.savefig(folder / "evaluation.png", dpi=150)
        plt.close(fig)
        merged = rows.merge(data[["id", "prior_mi"]], on="id", validate="one_to_one")
        subgroup = []
        for label, subset in [("No reported prior MI", merged.loc[merged.prior_mi.eq(0)]),
                              ("Prior MI", merged.loc[merged.prior_mi.eq(1)]),
                              ("Age 40–59", merged.loc[merged.age.lt(60)]),
                              ("Age 60–89", merged.loc[merged.age.ge(60)])]:
            subgroup.append({"group": label, "n": len(subset), "events": int(subset.target.sum()),
                             **weighted_metrics(subset.target, subset.probability, subset.weight, result["threshold"])})
        save_json(folder / "additional_subgroups.json", subgroup)
    for name, result in summary.items():
        ci = result["ci95_design_bootstrap"]
        counts = result["unweighted_screening"]
        baseline = result["weighted_constant_baseline"]
        lines += ["", f"## {name}", "", f"Weighted AUROC interval: {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f}; sensitivity interval: {ci['sensitivity'][0]:.1%}–{ci['sensitivity'][1]:.1%}; specificity interval: {ci['specificity'][0]:.1%}–{ci['specificity'][1]:.1%}.",
                  f"Unweighted screening counts: TP {counts['tp']}, FN {counts['fn']}, FP {counts['fp']}, TN {counts['tn']}. These counts should not be used to recompute the weighted percentages.",
                  f"Calibration method: {result['calibration']}; research screening threshold: {result['threshold']:.6f}. Constant baseline weighted accuracy: {baseline['accuracy']:.1%}, sensitivity zero, AUROC 0.5, Brier {baseline['brier']:.4f}."]
    lines += ["", "## Limits of the result", "",
              "The confidence intervals condition on this fitted model; they use rescaled pseudo-PSU resampling within strata and omit finite-population corrections. They do not cover training/search uncertainty or modern population shift. Predictions use a historical cohort that includes pre-existing disease. Public mortality causes/times are modified for some records to protect confidentiality. No external contemporary or Indian validation was performed.", "",
              "Accuracy at 0.5 is dominated by the much more common absence of a fatal event. The screening threshold intentionally makes a different sensitivity/specificity trade-off. Nonfatal events are outside this endpoint. The scores cannot be directly ranked against the minor project's existing-disease accuracy, the Framingham CHD benchmark, or unrelated IEEE diagnosis studies.", "",
              "This is a trained research system, not an industry-ready clinical model. Read ECG_RISK.md for ingestion, endpoint, validation and release requirements."]
    save_json(root / "summary.json", summary)
    save_json(root / "bundle_manifest.json", manifest)
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ECG run audited and packaged.")


if __name__ == "__main__":
    main()

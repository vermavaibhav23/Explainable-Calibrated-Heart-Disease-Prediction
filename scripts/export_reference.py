"""Export aggregate evidence only; raw data, identities and predictions remain local."""
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from heart_research.risk import save_json


def ecg_comparison(root):
    a = pd.read_csv(root / "routine" / "test_predictions.csv").sort_values("id").reset_index(drop=True)
    b = pd.read_csv(root / "routine_ecg" / "test_predictions.csv").sort_values("id").reset_index(drop=True)
    assert np.array_equal(a.id, b.id) and np.array_equal(a.target, b.target)
    groups = {s: {p: g.index.to_numpy() for p, g in frame.groupby("psu")} for s, frame in a.groupby("stratum")}
    assert all(len(g) == 2 for g in groups.values())
    rng = np.random.default_rng(2026)
    samples = []
    for _ in range(1000):
        idx = np.concatenate([g[k] for g in groups.values() for k in rng.choice(list(g), 1)])
        y, w = a.target.iloc[idx], a.weight.iloc[idx]
        samples.append([roc_auc_score(y, b.probability.iloc[idx], sample_weight=w) - roc_auc_score(y, a.probability.iloc[idx], sample_weight=w),
                        brier_score_loss(y, b.probability.iloc[idx], sample_weight=w) - brier_score_loss(y, a.probability.iloc[idx], sample_weight=w)])
    delta = {"direction": "routine_ecg minus routine", "interval_method": "paired rescaled pseudo-PSU bootstrap, conditional",
             "auc_difference": roc_auc_score(a.target, b.probability, sample_weight=a.weight) - roc_auc_score(a.target, a.probability, sample_weight=a.weight),
             "ci95_auc_difference": np.quantile(np.array(samples)[:, 0], [0.025, 0.975]).tolist(),
             "brier_difference": brier_score_loss(a.target, b.probability, sample_weight=a.weight) - brier_score_loss(a.target, a.probability, sample_weight=a.weight),
             "ci95_brier_difference": np.quantile(np.array(samples)[:, 1], [0.025, 0.975]).tolist()}
    save_json(root / "feature_set_comparison.json", delta)
    return delta


def export(root, target, result_doc):
    target.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "protocol.json").read_text())
    for field in ["splits", "partitions"]:
        if field in protocol:
            protocol[field + "_sha256"] = hashlib.sha256(json.dumps(protocol.pop(field), sort_keys=True).encode()).hexdigest()
    protocol["public_export"] = "Participant indices/IDs omitted; full local protocol retained. Reproduce using fixed seed, source checksum and code."
    save_json(target / "protocol.json", protocol)
    for name in ["summary.json", "bundle_manifest.json", "split_audit.json", "feature_set_comparison.json"]:
        if (root / name).exists():
            shutil.copy2(root / name, target / name)
    allow = {"selection.json", "development_search.json", "holdout_metrics.json", "test_metrics.json",
             "subgroups.json", "additional_subgroups.json", "missingness_stress.json", "decision_curve.csv",
             "results.json", "protocol.json"}
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.iterdir():
            if file.name in allow or file.name.startswith("search_") and file.suffix == ".csv":
                destination = target / folder.name / file.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, destination)
    shutil.copy2(root / "RESULTS.md", Path("docs") / result_doc)


def main():
    root = Path("artifacts/ecg-risk-v3")
    comparison = ecg_comparison(root)
    report = (root / "RESULTS.md").read_text(encoding="utf-8")
    marker = "\n## Paired ECG comparison and later-cohort check\n"
    report = report.split(marker)[0] + marker
    ci = comparison["ci95_auc_difference"]
    report += f"\nECG augmentation changed weighted AUROC by {comparison['auc_difference']:+.4f}, with conditional paired 95% interval {ci[0]:+.4f} to {ci[1]:+.4f}. This compares full selected pipelines, not a causal effect of ordering tests.\n"
    external = root / "external_2007" / "results.json"
    if external.exists():
        result = json.loads(external.read_text())
        m = result["weighted_screening"]
        report += f"\nThe frozen routine-only model was also evaluated without refitting, recalibration or threshold changes on {result['n']:,} NHANES 2007–2008 participants aged 40–79 ({result['events']} fatal events). Weighted AUROC {m['roc_auc']:.3f}; sensitivity {m['sensitivity']:.1%}; specificity {m['specificity']:.1%}; precision {m['precision']:.1%}; screening accuracy {m['accuracy']:.1%}; Brier {m['brier']:.4f}. Mean predicted risk {result['mean_predicted_risk']:.1%} versus observed risk {result['observed_risk']:.1%}. This is a later independent survey-cycle check of the routine model, not validation of the ECG model. Age restrictions, case mix and measurement differences prevent direct score ranking across cohorts.\n"
    (root / "RESULTS.md").write_text(report, encoding="utf-8")
    export(Path("artifacts/affordable-risk-v2"), Path("reports/affordable_risk"), "AFFORDABLE_RISK_RESULTS.md")
    export(root, Path("reports/ecg_risk"), "ECG_RESULTS.md")
    save_json(Path("reports/ecg_risk/environment.json"), {p: importlib.metadata.version(p) for p in
              ["numpy", "pandas", "scikit-learn", "catboost", "xgboost", "lightgbm", "shap", "streamlit", "fastapi"]})
    shutil.copy2("data/raw/nhanes3_risk_standardized.audit.json", "reports/ecg_risk/data_audit.json")
    print("Aggregate evidence exported; no patient records, participant predictions or model pickle files copied.")


if __name__ == "__main__":
    main()

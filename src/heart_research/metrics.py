import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def metrics(y, probability, threshold=0.5):
    y, p = np.asarray(y), np.asarray(probability)
    prediction = p >= np.asarray(threshold)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "positive_n": int(y.sum()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if both else None,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc_ap": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y, probability, minimum_sensitivity=0.90):
    """Maximize specificity subject to a development-only sensitivity target."""
    options = []
    for threshold in np.linspace(0.0, 1.0, 101):
        result = metrics(y, probability, threshold)
        if result["sensitivity"] is not None and result["sensitivity"] >= minimum_sensitivity:
            options.append((result["specificity"], threshold))
    return float(max(options)[1])


def bootstrap_intervals(y, probability, threshold, repeats=1000, seed=42):
    """Conditional percentile intervals; not repeated-training uncertainty."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(probability)
    thresholds = np.broadcast_to(threshold, y.shape)
    keys = ["accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc_ap", "brier"]
    samples = {k: [] for k in keys}
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        scores = metrics(y[idx], p[idx], thresholds[idx])
        for key in keys:
            if scores[key] is not None:
                samples[key].append(scores[key])
    return {
        key: np.quantile(values, [0.025, 0.975]).tolist() if values else None
        for key, values in samples.items()
    }

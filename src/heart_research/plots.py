from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def evaluation_plots(predictions, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    for kind in ["roc", "precision_recall", "calibration"]:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        for name, rows in predictions.groupby("model"):
            if kind == "roc":
                x, y, _ = roc_curve(rows.y, rows.probability)
            elif kind == "precision_recall":
                y, x, _ = precision_recall_curve(rows.y, rows.probability)
            else:
                y, x = calibration_curve(rows.y, rows.probability, n_bins=8, strategy="quantile")
            ax.plot(
                x,
                y,
                label=name.replace("_", " "),
                linewidth=1.6,
                marker="o" if kind == "calibration" else None,
            )
        labels = {
            "roc": ("False positive rate", "Sensitivity", "Nested out-of-fold ROC"),
            "precision_recall": ("Recall", "Precision", "Nested out-of-fold precision–recall"),
            "calibration": (
                "Mean predicted probability",
                "Observed positive fraction",
                "Reliability diagram",
            ),
        }
        ax.set(
            xlabel=labels[kind][0], ylabel=labels[kind][1], title=labels[kind][2], xlim=(0, 1), ylim=(0, 1)
        )
        if kind != "precision_recall":
            ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
        ax.legend(fontsize=8, loc="best")
        fig.savefig(Path(output) / f"{kind}.png", dpi=160)
        plt.close(fig)

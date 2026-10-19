import base64
import json
from html import escape
from pathlib import Path

import pandas as pd

STYLE = """body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#183039;padding:24px}
h1,h2{color:#0c6b68}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}
th,td{padding:9px;border-bottom:1px solid #d8e5e3;text-align:left}th{background:#edf6f4}
img{max-width:100%}.note{background:#edf6f4;padding:18px;border-radius:10px}small{color:#536b75}"""


def shell(title, body):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{escape(title)}</title><style>{STYLE}</style><body>{body}</body></html>'


def experiment_report(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    body = "<h1>Heart disease research · Experiment report</h1>"
    body += '<p class="note">Research prototype. Internal validation; not a diagnosis or future-event risk estimate.</p>'
    body += f"<p>{escape(manifest['protocol'])}</p>"
    body += f"<p>Records: {manifest['rows']} · Outer folds: {manifest['outer_folds']} · Seed: {manifest['seed']}</p>"
    if manifest["smoke_run"]:
        body += "<p><strong>SMOKE RUN: reduced protocol for software verification.</strong></p>"
    for name, title in [
        ("model_comparison", "Model comparison"),
        ("subgroups", "Subgroup evaluation"),
        ("robustness", "Missing-input stress tests"),
    ]:
        body += f"<h2>{title}</h2>" + pd.read_csv(folder / f"{name}.csv").to_html(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    intervals = json.loads((folder / "confidence_intervals.json").read_text())
    rows = [
        {"model": model, "metric": metric, "lower_95": bounds[0], "upper_95": bounds[1]}
        for model, values in intervals.items()
        for metric, bounds in values.items()
        if bounds
    ]
    body += "<h2>Conditional 95% bootstrap intervals</h2>" + pd.DataFrame(rows).to_html(index=False)
    for name in ["roc", "precision_recall", "calibration"]:
        encoded = base64.b64encode((folder / f"{name}.png").read_bytes()).decode()
        body += f'<img alt="{name}" src="data:image/png;base64,{encoded}">'
    body += (
        "<h2>Limitations</h2><ul>"
        + "".join(f"<li>{escape(s)}</li>" for s in manifest["limitations"])
        + "</ul>"
    )
    body += "<h2>Reproducibility manifest</h2><pre>" + escape(json.dumps(manifest, indent=2)) + "</pre>"
    (folder / "report.html").write_text(shell("Experiment report", body), encoding="utf-8")


def prediction_report(record, probability, threshold, explanation=None):
    body = "<h1>Heart disease classification · Research report</h1>"
    body += '<p class="note">Educational model output, not a diagnosis. No future-event time horizon is predicted.</p>'
    body += (
        f"<p>Model probability: <strong>{probability:.1%}</strong> · Decision threshold: {threshold:.2f}</p>"
    )
    body += f"<p>Model class: {int(probability >= threshold)}</p>"
    body += record.to_html(index=False)
    if explanation is not None:
        table = pd.DataFrame(
            {"Feature": explanation["features"], "Probability contribution": explanation["values"][0]}
        )
        body += "<h2>SHAP contributions</h2>" + table.to_html(index=False)
        body += f"<p>Background probability: {explanation['base_value']:.4f}. Contributions explain the model, not causes.</p>"
    return shell("Prediction report", body)

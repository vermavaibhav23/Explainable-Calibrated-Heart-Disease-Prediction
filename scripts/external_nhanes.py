"""Frozen routine-model evaluation on a later, separate NHANES 2007–2008 sample."""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import pandas as pd

from heart_research.ecg_risk import design_bootstrap, weighted_metrics
from heart_research.ecg_service import load
from heart_research.metrics import metrics
from heart_research.nhanes import ENDPOINT, ROUTINE, construct_target
from heart_research.risk import save_json

ROOT = Path("data/raw/nhanes2007")
RUN = Path("artifacts/ecg-risk-v3")
NEEDED = ["DEMO_E", "BPX_E", "BMX_E", "SMQ_E", "DIQ_E", "BPQ_E", "MCQ_E", "TCHOL_E", "HDL_E", "GHB_E", "BIOPRO_E"]


def download(item):
    name, url = item
    path = ROOT / name
    if not path.exists():
        with urlopen(url, timeout=90) as response:
            content = response.read()
        if content[:100].lstrip().startswith(b"<"):
            raise ValueError(f"Expected data, received HTML: {name}")
        path.write_bytes(content)
    return path


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    urls = {}
    for component in ["Demographics", "Examination", "Questionnaire", "Laboratory"]:
        page = f"https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={component}&Cycle=2007-2008"
        with urlopen(page, timeout=90) as response:
            html = response.read().decode("utf-8")
        for href in re.findall(r'href=["\']([^"\']+\.xpt)["\']', html, flags=re.I):
            stem = Path(href).stem.upper()
            if stem in NEEDED:
                urls[stem + ".xpt"] = urljoin(page, href)
    if len(urls) != len(NEEDED):
        raise ValueError(f"Missing official data links: {set(NEEDED)-{Path(k).stem for k in urls}}")
    urls["mortality.dat"] = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2007_2008_MORT_2019_PUBLIC.dat"
    save_json(ROOT / "sources.json", urls)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, urls.items()))
    columns = {"DEMO_E": ["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
               "BPX_E": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"],
               "BMX_E": ["SEQN", "BMXBMI"], "SMQ_E": ["SEQN", "SMQ020", "SMQ040"],
               "DIQ_E": ["SEQN", "DIQ010"], "BPQ_E": ["SEQN", "BPQ020", "BPQ040A", "BPQ050A"],
               "MCQ_E": ["SEQN", "MCQ160E"], "TCHOL_E": ["SEQN", "LBXTC"],
               "HDL_E": ["SEQN", "LBDHDD"], "GHB_E": ["SEQN", "LBXGH"], "BIOPRO_E": ["SEQN", "LBXSCR"]}
    frame = None
    for stem, selected in columns.items():
        part = pd.read_sas(ROOT / (stem + ".xpt"), format="xport")[selected]
        frame = part if frame is None else frame.merge(part, on="SEQN", how="left", validate="one_to_one")
    mortality = pd.read_fwf(ROOT / "mortality.dat", colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (45, 48)],
                             names=["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_EXM"], na_values=["."], dtype=float)
    frame = frame.merge(mortality, on="SEQN", how="left", validate="one_to_one")
    valid, y = construct_target(frame)
    valid &= frame.RIDAGEYR.between(40, 79) & frame.WTMEC2YR.gt(0)
    result = pd.DataFrame({"id": frame.SEQN, "age": frame.RIDAGEYR, "male": frame.RIAGENDR.map({1: 1, 2: 0}),
                           "weight": frame.WTMEC2YR, "stratum": frame.SDMVSTRA, "psu": frame.SDMVPSU,
                           "target": y, "bmi": frame.BMXBMI,
                           "systolic": frame[["BPXSY1", "BPXSY2", "BPXSY3"]].replace(0, np.nan).mean(axis=1),
                           "diastolic": frame[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan).mean(axis=1),
                           "total_cholesterol": frame.LBXTC, "hdl": frame.LBDHDD, "hba1c": frame.LBXGH,
                           "creatinine": frame.LBXSCR, "diabetes": frame.DIQ010.map({1: 1, 2: 0}),
                           "prior_mi": frame.MCQ160E.map({1: 1, 2: 0}),
                           "smoking": frame.SMQ040.map({1: 1, 2: 1, 3: 0}),
                           "bp_medication": frame.BPQ050A.map({1: 1, 2: 0})})
    result.loc[frame.SMQ020.eq(2), "smoking"] = 0
    result.loc[frame.BPQ020.eq(2) | frame.BPQ040A.eq(2), "bp_medication"] = 0
    # Released 2007 creatinine has already been standardized by CDC. Do not apply the NHANES III equation.
    result = result.loc[valid].reset_index(drop=True)
    result.to_csv(ROOT / "cohort.csv", index=False)
    return result


def main():
    output = RUN / "external_2007"
    output.mkdir(parents=True, exist_ok=False)
    # Fixed protocol is written before loading any external outcomes into evaluation.
    save_json(output / "protocol.json", {"cohort": "NHANES 2007–2008", "ages": "40–79 (80+ top-coded)",
                "endpoint": ENDPOINT, "frozen_model": "routine", "refitting": False, "recalibration": False,
                "threshold_retuned": False, "reason": "independent survey cycle with harmonizable routine inputs; no ECG validation"})
    frame = prepare()
    model, manifest = load(RUN, "routine")
    original = pd.read_csv("data/raw/nhanes3_risk_standardized.csv")
    hashes = pd.util.hash_pandas_object(frame[ROUTINE], index=False)
    if hashes.isin(pd.util.hash_pandas_object(original[ROUTINE], index=False)).any():
        raise ValueError("Exact baseline profile overlap; inspect before evaluation.")
    p = model.predict_proba(frame[ROUTINE])[:, 1]
    result = {"cohort": "NHANES 2007–2008, ages 40–79", "n": len(frame), "events": int(frame.target.sum()),
              "model_sha256": manifest["model_sha256"]["routine"], "endpoint": ENDPOINT,
              "threshold": model.threshold, "weighted_screening": weighted_metrics(frame.target, p, frame.weight, model.threshold),
              "weighted_at_0_5": weighted_metrics(frame.target, p, frame.weight, 0.5),
              "unweighted_screening": metrics(frame.target, p, model.threshold),
              "ci95_design_bootstrap": design_bootstrap(frame, p, model.threshold),
              "mean_predicted_risk": float(np.average(p, weights=frame.weight)),
              "observed_risk": float(np.average(frame.target, weights=frame.weight)),
              "missing_fraction": frame[ROUTINE].isna().mean().to_dict(),
              "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob("*") if f.suffix in [".xpt", ".dat"]},
              "limitations": ["Separate US survey cycle, not independent hospital/Indian validation",
                              "Ages limited to 40–79 due to public top-coding; different case mix",
                              "Measurement/assay differences and public mortality perturbation",
                              "Does not validate the ECG-augmented model"]}
    save_json(output / "results.json", result)
    frame["probability"] = p
    frame.to_csv(output / "predictions.csv", index=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ["files_sha256", "missing_fraction"]}, indent=2))


if __name__ == "__main__":
    main()

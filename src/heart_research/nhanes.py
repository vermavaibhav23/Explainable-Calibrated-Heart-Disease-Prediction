"""Official NHANES III fixed-width ingestion and explicit 10-year CVD-death endpoint."""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .risk import save_json

ROUTINE = ["age", "male", "systolic", "diastolic", "bmi", "smoking", "diabetes",
           "bp_medication", "prior_mi", "total_cholesterol", "hdl", "hba1c", "creatinine"]
ECG = ["ecg_rate", "pr_ms", "qrs_ms", "qt_ms", "p_axis", "qrs_axis", "t_axis", "major_ecg_abnormality"]
SETS = {"routine": ROUTINE, "routine_ecg": ROUTINE + ECG}
ENDPOINT = "10-year death from heart or cerebrovascular disease"


def read_fixed(root, stem, columns):
    text = (root / f"{stem}.sas").read_text(encoding="utf-8-sig")
    block = re.search(r"\bINPUT\s*\n(.*?);", text, flags=re.S | re.I).group(1)
    specs = {}
    for match in re.finditer(r"^\s*(\w+)\s+\$?(\d+)(?:-(\d+))?\s*$", block, flags=re.M):
        name, start, end = match.groups()
        specs[name] = (int(start) - 1, int(end or start))
    absent = set(columns) - specs.keys()
    if absent:
        raise ValueError(f"No fixed-width specification for {absent}")
    # Parse only documented selected variables, never execute the downloaded SAS.
    return pd.read_fwf(root / f"{stem}.dat", colspecs=[specs[c] for c in columns], names=columns,
                       na_values=["."], dtype=float)


def construct_target(frame, horizon=120):
    eligible = frame.ELIGSTAT.eq(1) & frame.MORTSTAT.isin([0, 1]) & frame.PERMTH_EXM.gt(0)
    death = frame.MORTSTAT.eq(1)
    early = frame.PERMTH_EXM.le(horizon)
    known_cause = frame.UCOD_LEADING.between(1, 10)
    # A competing death before ten years is a known non-CVD outcome for cumulative incidence.
    observed = frame.PERMTH_EXM.ge(horizon) | (death & known_cause)
    valid = eligible & observed & ~(death & early & ~known_cause)
    target = (death & early & frame.UCOD_LEADING.isin([1, 5])).astype(int)
    return valid, target


def prepare(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    ecg_columns = ["SEQN", "HSAGEIR", "HSSEX", "ECPRATE", "ECPPR", "ECPQRS", "ECPQT",
                   "ECPAXIS1", "ECPAXIS2", "ECPAXIS3", "ECPG1"]
    frame = read_fixed(root, "nh3ecg", ecg_columns)
    adult = read_fixed(root, "adult", ["SEQN", "DMPFSEQ", "SDPPHASE", "SDPPSU1", "SDPSTRA1", "SDPPSU2", "SDPSTRA2",
                                       "WTPFEX1", "WTPFEX2", "HAR1", "HAR3", "HAD1", "HAF10",
                                       "HAE2", "HAE4A", "HAE5A"])
    exam = read_fixed(root, "exam", ["SEQN", "PEPMNK1R", "PEPMNK5R", "BMPBMI"])
    lab = read_fixed(root, "lab", ["SEQN", "TCP", "HDP", "GHP", "CEP"])
    mort = pd.read_fwf(root / "mortality.dat", colspecs=[(0, 6), (14, 15), (15, 16), (16, 19), (45, 48)],
                        names=["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "PERMTH_EXM"],
                        na_values=["."], dtype=float)
    for other in [adult, exam, lab, mort]:
        frame = frame.merge(other, on="SEQN", how="left", validate="one_to_one")
    valid, target = construct_target(frame)
    valid &= frame.HSAGEIR.between(40, 89) & frame.SDPPHASE.isin([1, 2]) & frame.DMPFSEQ.notna()
    result = pd.DataFrame({"id": frame.SEQN, "family": frame.DMPFSEQ, "phase": frame.SDPPHASE,
                           "stratum": np.where(frame.SDPPHASE.eq(1), frame.SDPSTRA1, frame.SDPSTRA2),
                           "psu": np.where(frame.SDPPHASE.eq(1), frame.SDPPSU1, frame.SDPPSU2),
                           "weight": np.where(frame.SDPPHASE.eq(1), frame.WTPFEX1, frame.WTPFEX2),
                           "target": target, "followup_months": frame.PERMTH_EXM,
                           "age": frame.HSAGEIR, "male": frame.HSSEX.map({1: 1, 2: 0})})
    mapping = {"PEPMNK1R": "systolic", "PEPMNK5R": "diastolic", "BMPBMI": "bmi",
               "TCP": "total_cholesterol", "HDP": "hdl", "GHP": "hba1c", "CEP": "creatinine",
               "ECPRATE": "ecg_rate", "ECPPR": "pr_ms", "ECPQRS": "qrs_ms", "ECPQT": "qt_ms",
               "ECPAXIS1": "p_axis", "ECPAXIS2": "qrs_axis", "ECPAXIS3": "t_axis"}
    for source, name in mapping.items():
        # Documented special nonresponse codes; retain ordinary observed extremes.
        result[name] = frame[source].replace([888, 8888, 88888, 888.8, 8888.8, 888.88], np.nan)
    # CDC laboratory documentation, 2006 revision, p111: standardize historical assay.
    result["creatinine"] = 0.960 * result["creatinine"] - 0.184
    for source, name in [("HAR3", "smoking"), ("HAD1", "diabetes"), ("HAF10", "prior_mi"),
                         ("HAE5A", "bp_medication")]:
        result[name] = frame[source].map({1: 1, 2: 0})
    result.loc[frame.HAR1.eq(2), "smoking"] = 0
    result.loc[frame.HAE2.eq(2) | frame.HAE4A.eq(2), "bp_medication"] = 0
    # ECG flag is 0=no, 1=yes (different from household yes/no coding).
    result["major_ecg_abnormality"] = frame.ECPG1.where(frame.ECPG1.isin([0, 1]))
    valid &= result.weight.gt(0) & result.stratum.notna() & result.psu.notna()
    result = result.loc[valid].reset_index(drop=True)
    if not result.id.is_unique:
        raise ValueError("Repeated participants found.")
    for name in ["id", "family", "phase", "stratum", "psu"]:
        result[name] = result[name].astype(int)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    audit = {"endpoint": ENDPOINT, "source_ecg_rows": len(frame), "retained_rows": len(result),
              "excluded_rows": int((~valid).sum()), "events": int(result.target.sum()),
              "phase_rows": result.phase.value_counts().to_dict(),
              "missing_fraction": result[ROUTINE + ECG].isna().mean().to_dict(),
              "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*")
                               if p.suffix in [".sas", ".dat"]}}
    save_json(output.with_suffix(".audit.json"), audit)
    print(audit, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/nhanes3")
    parser.add_argument("--output", default="data/raw/nhanes3_risk_standardized.csv")
    args = parser.parse_args()
    prepare(args.raw, args.output)

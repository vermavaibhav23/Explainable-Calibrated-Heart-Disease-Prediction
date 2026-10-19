import numpy as np
import pandas as pd

from heart_research.ecg_risk import weighted_threshold
from heart_research.nhanes import SETS, construct_target, read_fixed


def test_endpoint_handles_censoring_competing_death_and_unknown_cause():
    frame = pd.DataFrame({"ELIGSTAT": [1, 1, 1, 1, 1, 3, 1, 1],
                          "MORTSTAT": [1, 1, 0, 0, 1, 1, 1, 1],
                          "PERMTH_EXM": [60, 60, 100, 200, 60, 60, 130, 0],
                          "UCOD_LEADING": [1, 2, np.nan, np.nan, np.nan, 5, 5, 1]})
    valid, y = construct_target(frame)
    assert valid.tolist() == [True, True, False, True, False, False, True, False]
    assert y[valid].tolist() == [1, 0, 0, 0]


def test_fixed_width_spec_is_read_not_executed(tmp_path):
    (tmp_path / "sample.sas").write_text('INPUT\n SEQN 1-3\n AGE 4-5\n;\n arbitrary code;', encoding="utf-8")
    (tmp_path / "sample.dat").write_text("00150\n00265\n", encoding="utf-8")
    result = read_fixed(tmp_path, "sample", ["SEQN", "AGE"])
    assert result.AGE.tolist() == [50, 65]


def test_followup_and_design_fields_cannot_enter_model():
    forbidden = {"target", "followup_months", "id", "family", "phase", "weight", "stratum", "psu", "MORTSTAT"}
    assert all(not forbidden.intersection(fields) for fields in SETS.values())


def test_screening_target_is_weighted():
    y = np.array([1, 1, 1, 0])
    p = np.array([0.1, 0.5, 0.9, 0.2])
    w = np.array([9, 1, 1, 5])
    threshold = weighted_threshold(y, p, w)
    assert np.average(p[y == 1] >= threshold, weights=w[y == 1]) >= 0.8

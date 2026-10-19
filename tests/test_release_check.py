import numpy as np
import pytest

from heart_research.release_check import acceptance, threshold_envelope


def test_majority_accuracy_cannot_pass_joint_gate():
    y = np.array([0] * 94 + [1] * 6)
    result = threshold_envelope(y, np.zeros(100), np.ones(100))
    assert not result["joint_target_achievable_by_any_threshold_on_this_sample"]
    assert result["maximum_sensitivity_at_required_accuracy"] == 0
    assert result["maximum_accuracy_at_required_sensitivity"] == pytest.approx(0.06)
    assert not acceptance({"accuracy": 0.94, "sensitivity": 0})


def test_perfect_predictions_pass_both_targets():
    result = threshold_envelope([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], [1, 2, 3, 4])
    assert result["joint_target_achievable_by_any_threshold_on_this_sample"]
    assert acceptance({"accuracy": 0.93, "sensitivity": 0.9})


@pytest.mark.parametrize("metrics", [{}, {"accuracy": float("nan"), "sensitivity": 1}, {"accuracy": True, "sensitivity": 1}])
def test_invalid_evidence_fails_closed(metrics):
    assert not acceptance(metrics)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError, match="positive"):
        threshold_envelope([0, 1], [0.1, 0.9], [1, -1])

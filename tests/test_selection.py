"""D18–D21's selection and horizon rules, checked on scripted curves rather than training."""

from __future__ import annotations

import pytest

from temporallens.training.selection import (
    ESCALATED_HORIZON_EPOCHS,
    INITIAL_HORIZON_EPOCHS,
    SELECTION_WINDOW_EPOCHS,
    common_budget,
    decide_horizon,
    fold_score,
    select_epoch,
    smoothed_selection_scores,
)


def test_the_fold_score_weighs_subjects_equally_whatever_their_window_counts() -> None:
    """D21: the mean of four subject scores, never a pooled-window score."""
    assert fold_score({1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="exactly 4"):
        fold_score({1: 0.2, 2: 0.4, 3: 0.6})


def test_epochs_before_a_full_window_are_ineligible() -> None:
    """Epoch 9's trailing window would be incomplete, so it cannot be selected however high."""
    curve = [0.9] * 9 + [0.1] * 21
    smoothed = smoothed_selection_scores(curve)
    assert smoothed[: SELECTION_WINDOW_EPOCHS - 1] == [None] * 9
    assert smoothed[9] == pytest.approx(0.82)
    assert select_epoch(curve).epoch == SELECTION_WINDOW_EPOCHS


def test_the_smoothed_peak_not_the_raw_spike_is_selected() -> None:
    """One noisy spike does not win against a sustained plateau (§5.2)."""
    curve = [0.5] * 30
    curve[11] = 0.99  # a single raw spike at epoch 12
    for epoch in range(20, 31):
        curve[epoch - 1] = 0.6
    assert select_epoch(curve).epoch == 29


def test_an_exact_tie_goes_to_the_earliest_epoch() -> None:
    curve = [0.3] * 30
    selection = select_epoch(curve)
    assert selection.epoch == SELECTION_WINDOW_EPOCHS
    assert selection.smoothed_score == pytest.approx(0.3)


def test_the_common_budget_is_the_ceiling_of_the_median() -> None:
    """D18: ceil((e(4) + e(5)) / 2) over the eight sorted fold epochs."""
    assert common_budget([10, 12, 13, 14, 17, 20, 25, 30]) == 16  # ceil(15.5)
    assert common_budget([11, 11, 11, 12, 12, 30, 30, 30]) == 12  # ceil(12.0)
    with pytest.raises(ValueError, match="expected 8"):
        common_budget([10, 11, 12])


def test_d19_escalates_once_then_blocks() -> None:
    assert decide_horizon(29, INITIAL_HORIZON_EPOCHS).action == "accept"
    escalate = decide_horizon(INITIAL_HORIZON_EPOCHS, INITIAL_HORIZON_EPOCHS)
    assert escalate.action == "escalate"
    assert escalate.next_horizon == ESCALATED_HORIZON_EPOCHS
    assert decide_horizon(59, ESCALATED_HORIZON_EPOCHS).action == "accept"
    assert decide_horizon(ESCALATED_HORIZON_EPOCHS, ESCALATED_HORIZON_EPOCHS).action == "block"


def test_a_horizon_or_budget_outside_the_protocol_is_refused() -> None:
    with pytest.raises(ValueError, match="horizon must be"):
        decide_horizon(10, 45)
    with pytest.raises(ValueError, match="impossible"):
        decide_horizon(9, INITIAL_HORIZON_EPOCHS)


def test_a_curve_with_an_invalid_score_is_refused() -> None:
    with pytest.raises(ValueError, match="finite values in"):
        select_epoch([0.5] * 29 + [float("nan")])

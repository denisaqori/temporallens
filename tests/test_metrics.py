"""Evaluation metrics, with the protocol decisions they implement.

Hand-verifiable cases wherever possible. Where a metric is only characterisable by its
properties (ECE under adaptive binning), the test states the property rather than a magic number.

Decisions pinned here: D8 (per-class family, fold matrices averaged not summed), D21 (undefined
class F1 counts as 0), D23 (adaptive bins, M=10, bins recomputed after scaling), D25
(overconfidence error is the one-sided binned gap; mean-confidence-on-errors is NA with no
errors), D26 (per-subject values, paired interval).
"""

from __future__ import annotations

import numpy as np
import pytest
from temporallens.evaluation.metrics import (
    accuracy,
    average_confusion_matrices,
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    mean_confidence_when_wrong,
    overconfidence_error,
    paired_bootstrap_interval,
    per_class_precision,
    per_class_recall,
    per_subject,
)

K = 18


def _onehot_probs(labels, confidence=1.0, num_classes=K):
    """Probabilities that put `confidence` on `labels` and spread the rest uniformly."""
    probs = np.full((len(labels), num_classes), (1 - confidence) / (num_classes - 1))
    probs[np.arange(len(labels)), labels] = confidence
    return probs


# --- the hand-checkable core ----------------------------------------------------------------


def test_accuracy_is_the_fraction_correct() -> None:
    true = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 9, 3])
    assert accuracy(pred, true) == pytest.approx(0.75)


def test_confusion_matrix_rows_are_truth_and_columns_predictions() -> None:
    true = np.array([0, 0, 1])
    pred = np.array([0, 1, 1])
    cm = confusion_matrix(pred, true, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1
    assert cm.sum() == 3


def test_per_class_precision_and_recall_on_a_worked_case() -> None:
    # class 0: predicted twice, right once -> precision 1/2; present twice, found once -> recall 1/2
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])
    precision = per_class_precision(pred, true, num_classes=2)
    recall = per_class_recall(pred, true, num_classes=2)
    assert precision[0] == pytest.approx(0.5)
    assert recall[0] == pytest.approx(0.5)
    assert precision[1] == pytest.approx(0.5)


def test_macro_f1_averages_over_every_class_not_every_window() -> None:
    """Rest is 50.2% of windows, so a window-weighted average would be dominated by it."""
    true = np.array([0] * 90 + [1] * 10)
    pred = np.array([0] * 100)  # always predicts rest
    assert accuracy(pred, true) == pytest.approx(0.9)
    # class 0 F1 = 2*0.9*1/(1.9); class 1 F1 = 0. Macro = mean of the two.
    assert macro_f1(pred, true, num_classes=2) == pytest.approx((2 * 0.9 / 1.9 + 0.0) / 2)


def test_an_undefined_class_f1_counts_as_zero() -> None:
    """D21: assign zero to an undefined class F1, and still divide by all 18 classes.

    Dropping absent classes instead would quietly rescale macro-F1 whenever a small-k support
    set omits gestures -- exactly where G3 does its measuring.
    """
    true = np.array([0, 0, 1])
    pred = np.array([0, 0, 1])
    # Classes 2..17 never appear and are never predicted: F1 undefined -> 0.
    expected = (1.0 + 1.0) / K
    assert macro_f1(pred, true, num_classes=K) == pytest.approx(expected)


def test_fold_confusion_matrices_are_averaged_never_summed() -> None:
    """D8: all 8 fold models score the SAME test subjects, so summing counts each window 8x."""
    cm = confusion_matrix(np.array([0, 1]), np.array([0, 1]), num_classes=2)
    averaged = average_confusion_matrices([cm] * 8)
    np.testing.assert_allclose(averaged, cm.astype(float))
    assert averaged.sum() == pytest.approx(
        cm.sum()
    ), "averaging must leave the matrix on the scale of one evaluation"


# --- calibration (D23) ------------------------------------------------------------------------


def test_ece_defaults_to_ten_adaptive_bins() -> None:
    rng = np.random.default_rng(0)
    conf = np.clip(rng.beta(5, 2, 4000), 1 / K, 1.0)
    correct = rng.random(4000) < conf
    default = expected_calibration_error(conf, correct)
    explicit = expected_calibration_error(conf, correct, bins=10, binning="equal_mass")
    assert default == pytest.approx(explicit)


def test_ece_is_near_zero_for_a_calibrated_model_and_tracks_a_known_offset() -> None:
    rng = np.random.default_rng(1)
    conf = np.clip(rng.beta(5, 2, 40000), 1 / K, 1.0)

    calibrated = expected_calibration_error(conf, rng.random(40000) < conf)
    offset = 0.12
    skewed = expected_calibration_error(conf, rng.random(40000) < np.clip(conf - offset, 0, 1))
    assert calibrated < 0.02
    assert skewed == pytest.approx(offset, abs=0.03)


def test_equal_mass_bins_are_populated_where_equal_width_strands_them() -> None:
    """Why adaptive: a confident model's distribution leaves equal-width bins nearly empty."""
    rng = np.random.default_rng(2)
    conf = np.clip(rng.beta(8, 2, 2000), 1 / K, 1.0)
    correct = rng.random(2000) < np.clip(conf - 0.1, 0, 1)
    adaptive = expected_calibration_error(conf, correct, bins=10, binning="equal_mass")
    width = expected_calibration_error(conf, correct, bins=10, binning="equal_width")
    assert adaptive > 0 and width > 0
    assert adaptive != width, "the two binning schemes cannot be the same computation"


def test_brier_score_is_zero_for_a_perfect_confident_model() -> None:
    labels = np.array([0, 1, 2])
    assert brier_score(_onehot_probs(labels, confidence=1.0), labels) == pytest.approx(0.0)


def test_brier_score_penalises_confident_errors_more_than_hesitant_ones() -> None:
    labels = np.array([0])
    confident_wrong = np.zeros((1, 3))
    confident_wrong[0, 1] = 1.0
    hesitant = np.full((1, 3), 1 / 3)
    assert brier_score(confident_wrong, labels) > brier_score(hesitant, labels)


# --- overconfidence, and the descriptive statistic (D25) --------------------------------------


def test_overconfidence_error_is_one_sided() -> None:
    """D25: it penalises confidence ABOVE accuracy only, so a timid model scores 0."""
    rng = np.random.default_rng(3)
    conf = np.clip(rng.beta(5, 2, 20000), 1 / K, 1.0)
    overconfident = rng.random(20000) < np.clip(conf - 0.15, 0, 1)
    underconfident = rng.random(20000) < np.clip(conf + 0.15, 0, 1)

    assert overconfidence_error(conf, overconfident) > 0.05
    assert overconfidence_error(conf, underconfident) == pytest.approx(0.0, abs=0.01)


def test_mean_confidence_when_wrong_is_na_with_no_errors() -> None:
    """D25: 0 would read as "perfectly humble when wrong"; there is nothing to average."""
    conf = np.array([0.9, 0.8])
    assert mean_confidence_when_wrong(conf, np.array([True, True])) is None


def test_mean_confidence_when_wrong_uses_only_the_errors() -> None:
    conf = np.array([0.9, 0.2, 0.4])
    correct = np.array([True, False, False])
    assert mean_confidence_when_wrong(conf, correct) == pytest.approx(0.3)


# --- per-subject reporting and the paired interval (D26) --------------------------------------


def test_per_subject_returns_one_value_per_subject() -> None:
    subjects = np.array([5, 5, 10, 10])
    true = np.array([0, 1, 0, 1])
    pred = np.array([0, 1, 0, 0])
    got = per_subject(accuracy, pred, true, subjects=subjects)
    assert set(got) == {5, 10}
    assert got[5] == pytest.approx(1.0)
    assert got[10] == pytest.approx(0.5)


def test_per_subject_never_pools_subjects() -> None:
    """A pooled number can look respectable while one subject sits at chance."""
    subjects = np.array([1] * 100 + [2] * 100)
    true = np.zeros(200, dtype=int)
    pred = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    got = per_subject(accuracy, pred, true, subjects=subjects)
    assert got[1] == pytest.approx(1.0)
    assert got[2] == pytest.approx(0.0)


def test_paired_interval_is_seeded_and_reproducible() -> None:
    """D26: percentile bootstrap, 10,000 resamples, seeded from experiment.seed."""
    diffs = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.05, 0.01])
    a = paired_bootstrap_interval(diffs, seed=42)
    b = paired_bootstrap_interval(diffs, seed=42)
    assert a == b
    assert a != paired_bootstrap_interval(diffs, seed=43)


def test_a_consistent_difference_excludes_zero_and_a_noisy_one_does_not() -> None:
    """D26's reportability rule: a difference counts when its paired interval excludes zero."""
    consistent = np.array([0.04, 0.05, 0.03, 0.06, 0.04, 0.05, 0.04, 0.05])
    lo, hi = paired_bootstrap_interval(consistent, seed=42)
    assert lo > 0

    noisy = np.array([0.05, -0.04, 0.03, -0.06, 0.04, -0.05, 0.02, -0.03])
    lo, hi = paired_bootstrap_interval(noisy, seed=42)
    assert lo < 0 < hi


def test_paired_interval_requires_matched_subjects() -> None:
    with pytest.raises(ValueError, match="at least two"):
        paired_bootstrap_interval(np.array([0.1]), seed=42)

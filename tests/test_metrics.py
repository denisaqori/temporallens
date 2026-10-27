"""Evaluation metrics, with the protocol decisions they implement.

Hand-verifiable cases wherever possible. Where a metric is only characterisable by its
properties (ECE under adaptive binning), the test states the property rather than a magic number.

Decisions pinned here: D8 (per-class family, fold matrices averaged not summed), D21 (undefined
class F1 counts as 0), D23 (adaptive bins, M=10, bins recomputed after scaling), D25
(overconfidence error is the one-sided binned gap; mean-confidence-on-errors is NA with no
errors), D26 (per-subject values, paired interval), D39 (unhalved Brier and confidence ties).
"""

from __future__ import annotations

import numpy as np
import pytest

from temporallens.evaluation.metrics import (
    _bin_edges,
    accuracy,
    average_confusion_matrices,
    brier_score,
    calibration_bins,
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


def test_brier_score_is_two_for_a_certain_wrong_prediction() -> None:
    """D39: full multiclass vector, not the binary or half-normalized convention."""
    labels = np.array([0])
    certain_wrong = _onehot_probs(np.array([1]), confidence=1.0)
    assert brier_score(certain_wrong, labels) == pytest.approx(2.0)


def test_brier_score_sums_classes_then_averages_windows() -> None:
    # Three-class examples: the first has error .04 + .01 + .01; the second has error 2.
    # Neither dividing by 3 classes nor halving the vector error is permitted.
    probabilities = np.array([[0.8, 0.1, 0.1], [0.0, 1.0, 0.0]])
    labels = np.array([0, 0])
    assert brier_score(probabilities, labels) == pytest.approx((0.06 + 2.0) / 2)


def test_equal_mass_confidence_ties_stay_in_one_nonempty_bin() -> None:
    conf = np.full(100, 0.5)
    correct = np.array([True] * 50 + [False] * 50)
    formed = calibration_bins(conf, correct)
    assert (formed.nominal_count, formed.effective_count) == (10, 1)
    assert formed.weight.tolist() == [1.0]
    assert (formed.mean_confidence.tolist(), formed.accuracy.tolist()) == ([0.5], [0.5])
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)
    # Splitting these ties by input order would manufacture miscalibration.
    permutation = np.random.default_rng(7).permutation(len(conf))
    assert expected_calibration_error(conf[permutation], correct[permutation]) == pytest.approx(0.0)


def test_equal_mass_quantiles_collapse_duplicate_cutpoints_without_losing_windows() -> None:
    conf = np.array([0.2] * 20 + [0.8] * 80)
    correct = np.array([True] * 10 + [False] * 10 + [True] * 70 + [False] * 10)
    edges = _bin_edges(conf, bins=10, binning="equal_mass")
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == 0.0 and edges[-1] == 1.0
    formed = calibration_bins(conf, correct)
    assert formed.effective_count == 2
    assert int(formed.occupancy.sum()) == conf.size, "collapsing cutpoints must not drop windows"
    assert formed.weight.tolist() == pytest.approx([0.2, 0.8])
    assert expected_calibration_error(conf, correct) == pytest.approx(0.12)


def test_equal_mass_boundaries_are_linear_and_equal_cutpoints_go_to_upper_bin() -> None:
    conf = np.array([0.1, 0.3, 0.5, 0.9])
    np.testing.assert_allclose(_bin_edges(conf, bins=2, binning="equal_mass"), [0.0, 0.4, 1.0])
    tied_at_cut = np.array([0.0, 0.5, 0.5, 1.0])
    formed = calibration_bins(tied_at_cut, np.ones(4, dtype=bool), bins=2)
    assert formed.weight.tolist() == pytest.approx([0.25, 0.75])
    assert formed.lower.tolist() == pytest.approx([0.0, 0.5])


def test_ece_and_overconfidence_are_computed_from_the_bins_a_report_would_show() -> None:
    """D39 requires the bin occupancy to be reported beside the metric.

    If the reporter formed its own bins, the occupancy it printed and the ECE beside it could
    come from two different binnings. Both metrics are therefore defined over the same object a
    report persists, and this pins that they agree with it exactly.
    """
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.3, 1.0, 500)
    correct = rng.uniform(size=500) < conf - 0.1  # overconfident by construction
    formed = calibration_bins(conf, correct)
    gap = formed.accuracy - formed.mean_confidence
    assert expected_calibration_error(conf, correct) == pytest.approx(
        float(np.sum(formed.weight * np.abs(gap)))
    )
    assert overconfidence_error(conf, correct) == pytest.approx(
        float(np.sum(formed.weight * formed.mean_confidence * np.maximum(-gap, 0.0)))
    )
    assert formed.effective_count == 10, "continuous confidences should fill every nominal bin"


def test_saturated_confidences_collapse_bins_and_the_report_says_so() -> None:
    """In float32 softmax rounds to exactly 1.0 once the logit gap passes about 17, so a confident
    model can tie most of its windows at 1.0. The bins must collapse rather than split those ties
    by example order, and the reportable form must carry the effective count."""
    conf = np.concatenate([np.full(70, 1.0), np.linspace(0.4, 0.95, 30)])
    correct = np.ones(conf.size, dtype=bool)
    formed = calibration_bins(conf, correct)
    assert formed.effective_count < formed.nominal_count
    assert int(formed.occupancy.sum()) == conf.size
    report = formed.to_dict()
    assert report["nominal_count"] == 10 and report["effective_count"] == formed.effective_count
    import json

    assert json.loads(json.dumps(report)) == report, "the bin report must be JSON-serialisable"


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


# --- inputs are checked, not trusted ----------------------------------------------------------
# Each case below used to be accepted: NumPy either failed somewhere unrelated or returned a
# plausible number. A confidence of 1.7 produced an ECE of 0.33; a label of -1 was counted as
# class 17, because negative indices wrap around.


def test_a_negative_label_is_refused_rather_than_counted_as_the_last_class() -> None:
    with pytest.raises(ValueError, match=r"labels must lie in \[0, 18\)"):
        confusion_matrix(np.array([0, 1]), np.array([0, -1]), num_classes=18)


@pytest.mark.parametrize(
    ("predicted", "true", "reason"),
    [
        (np.array([0, 1, 1]), np.array([0, 1]), "same length"),
        (np.array([[0, 1]]), np.array([[0, 1]]), "1-D"),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64), "empty"),
        (np.array([0.5, 1.0]), np.array([0, 1]), "integer"),
        (np.array([True, False]), np.array([0, 1]), "integer"),
    ],
)
def test_malformed_labels_are_refused(predicted, true, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        accuracy(predicted, true)
    with pytest.raises(ValueError, match=reason):
        macro_f1(predicted, true, num_classes=18)


@pytest.mark.parametrize(
    ("probabilities", "true", "reason"),
    [
        (np.array([[0.9, 0.9], [0.1, 0.1]]), np.array([0, 1]), "sum to 1"),
        (np.array([[1.2, -0.2], [0.5, 0.5]]), np.array([0, 1]), r"\[0, 1\]"),
        (np.array([[np.nan, 1.0], [0.5, 0.5]]), np.array([0, 1]), "finite"),
        (np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0, 1, 1]), "one row per label"),
        (np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0, 2]), r"labels must lie in \[0, 2\)"),
        (np.array([0.5, 0.5]), np.array([0]), "2-D"),
    ],
)
def test_malformed_probabilities_are_refused(probabilities, true, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        brier_score(probabilities, true)


def test_float32_softmax_rows_are_accepted() -> None:
    """The tolerance has to admit what the trainer actually produces."""
    logits = np.random.default_rng(0).normal(size=(500, 18)).astype(np.float32)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    brier_score(probabilities.astype(np.float64), np.zeros(500, dtype=np.int64))


@pytest.mark.parametrize(
    ("confidence", "correct", "reason"),
    [
        (np.array([1.7, 0.2, 0.9]), np.array([True, False, True]), r"\[0, 1\]"),
        (np.array([-0.1, 0.2, 0.9]), np.array([True, False, True]), r"\[0, 1\]"),
        (np.array([np.nan, 0.2, 0.9]), np.array([True, False, True]), "finite"),
        (np.array([0.5, 0.2]), np.array([True, False, True]), "same length"),
        (np.array([0.5, 0.2]), np.array([1, 2]), "true or false"),
        (np.array([], dtype=np.float64), np.array([], dtype=bool), "empty"),
    ],
)
def test_malformed_confidences_are_refused(confidence, correct, reason) -> None:
    for metric in (expected_calibration_error, overconfidence_error, mean_confidence_when_wrong):
        with pytest.raises(ValueError, match=reason):
            metric(confidence, correct)
    with pytest.raises(ValueError, match=reason):
        calibration_bins(confidence, correct)


def test_a_bin_count_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="bins"):
        calibration_bins(np.array([0.5, 0.6]), np.array([True, False]), bins=0)


def test_per_subject_refuses_subjects_that_do_not_line_up_with_the_windows() -> None:
    with pytest.raises(ValueError, match="one subject per window"):
        per_subject(accuracy, np.array([0, 1, 1]), np.array([0, 1, 0]), subjects=np.array([1, 2]))


@pytest.mark.parametrize(
    ("differences", "kwargs", "reason"),
    [
        (np.array([0.1, np.nan, 0.2]), {}, "finite"),
        (np.array([0.1, 0.2, 0.3]), {"resamples": 0}, "resamples"),
        (np.array([0.1, 0.2, 0.3]), {"confidence": 1.0}, "confidence"),
    ],
)
def test_the_paired_interval_refuses_malformed_inputs(differences, kwargs, reason) -> None:
    with pytest.raises(ValueError, match=reason):
        paired_bootstrap_interval(differences, seed=0, **kwargs)

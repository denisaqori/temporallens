"""Evaluation metrics, and the protocol decisions that define them.

Most of these are ordinary, but four carry decisions that are easy to get subtly wrong:

* ``macro_f1`` divides by **every** class, counting an undefined class as 0 (D21). Dropping absent
  classes instead would quietly rescale the metric whenever a small-*k* support set omits
  gestures — exactly where G3 does its measuring.
* ``average_confusion_matrices`` averages, never sums (D8). All eight fold models score the *same*
  test subjects, so summing counts every test window eight times.
* ``expected_calibration_error`` uses adaptive (equal-mass) bins, nominally M=10 (D23). Identical
  confidences always share a bin, so ties can leave fewer bins occupied (D39). A perfectly
  calibrated model still scores above 0 at finite sample size; that threshold depends on the
  evaluation design, so it is recomputed per design and never subtracted (D33).
* ``calibration_bins`` is the one place bins are formed. ECE, the overconfidence error and any
  report of bin occupancy all read it, so a reported number and the bins it came from cannot
  disagree.
* ``overconfidence_error`` is the one-sided, confidence-weighted positive gap (D25). A uniformly
  timid model scores a perfect 0 on it while being badly miscalibrated, so it is only
  interpretable beside ECE. ``mean_confidence_when_wrong`` is the separate *descriptive*
  statistic and returns ``None``, never 0, when a model makes no errors.

Uncertainty resamples **subjects**, never windows (D26): windows overlap 75% at stride 100, and a
window-level bootstrap produced an interval 20x too narrow on identical data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

#: D23's bin count, and the binning scheme the floor measurements assume.
DEFAULT_ECE_BINS = 10
DEFAULT_BINNING = "equal_mass"
#: D26's paired interval.
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def accuracy(predicted: IntArray, true: IntArray) -> float:
    """Fraction of windows whose predicted class matches the truth.

    Misleading alone under this project's imbalance: rest is ~50% of windows, so a model
    predicting only rest scores 0.5. `macro_f1` is the headline for that reason.
    """
    return float(np.mean(np.asarray(predicted) == np.asarray(true)))


def confusion_matrix(predicted: IntArray, true: IntArray, *, num_classes: int) -> IntArray:
    """Rows are truth, columns are predictions."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(true), np.asarray(predicted)), 1)
    return matrix


def average_confusion_matrices(matrices: Sequence[npt.NDArray[np.number]]) -> FloatArray:
    """Element-wise mean, which keeps the matrix on the scale of one evaluation (D8)."""
    if not matrices:
        raise ValueError("cannot average an empty list of confusion matrices")
    return np.mean(np.stack([np.asarray(m, dtype=np.float64) for m in matrices]), axis=0)


def _true_false_counts(
    predicted: IntArray, true: IntArray, num_classes: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    matrix = confusion_matrix(predicted, true, num_classes=num_classes).astype(np.float64)
    true_positive = np.diag(matrix)
    return true_positive, matrix.sum(axis=0) - true_positive, matrix.sum(axis=1) - true_positive


def per_class_precision(predicted: IntArray, true: IntArray, *, num_classes: int) -> FloatArray:
    """Precision per class: when this gesture is predicted, how often is it right?

    Catches classes the model over-predicts. A class never predicted scores 0, not NaN.
    """
    tp, fp, _ = _true_false_counts(predicted, true, num_classes)
    denominator = tp + fp
    return np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def per_class_recall(predicted: IntArray, true: IntArray, *, num_classes: int) -> FloatArray:
    """Recall per class: how much of this gesture does the model find?

    Catches classes it quietly misses, which accuracy hides and macro-F1 averages away.
    """
    tp, _, fn = _true_false_counts(predicted, true, num_classes)
    denominator = tp + fn
    return np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def per_class_f1(predicted: IntArray, true: IntArray, *, num_classes: int) -> FloatArray:
    """Per-class F1, with an undefined class scored 0 rather than dropped (D21)."""
    precision = per_class_precision(predicted, true, num_classes=num_classes)
    recall = per_class_recall(predicted, true, num_classes=num_classes)
    denominator = precision + recall
    return np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0
    )


def macro_f1(predicted: IntArray, true: IntArray, *, num_classes: int) -> float:
    """Unweighted mean over **all** ``num_classes`` classes (D21)."""
    return float(np.mean(per_class_f1(predicted, true, num_classes=num_classes)))


def brier_score(probabilities: FloatArray, true: IntArray) -> float:
    """Class-sum squared probability error, then window mean; unhalved [0, 2] (D39).

    A certain wrong prediction misses the true class and predicts another, contributing 1
    from each. Dividing by the class count or by 2 would implement a different convention.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    target = np.zeros_like(probs)
    target[np.arange(len(true)), np.asarray(true)] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def _bin_edges(confidence: FloatArray, bins: int, binning: str) -> FloatArray:
    if binning == "equal_width":
        return np.linspace(0.0, 1.0, bins + 1)
    if binning == "equal_mass":
        edges = np.quantile(confidence, np.linspace(0.0, 1.0, bins + 1), method="linear")
        edges[0], edges[-1] = 0.0, 1.0
        # D39: identical confidence values stay together; M is a nominal target, not a
        # reason to split ties by example order or labels. Empty intervals carry no weight.
        return np.unique(edges)
    raise ValueError(f"unknown binning {binning!r}; expected equal_mass or equal_width")


@dataclass(frozen=True)
class CalibrationBins:
    """The occupied confidence bins one calibration measurement is computed from (D23, D39).

    Only occupied bins are kept; ``nominal_count`` is the M that was asked for and
    ``effective_count`` how many bins actually hold windows. D39 requires both to be reported,
    because heavy confidence ties (for example softmax outputs that round to exactly 1.0) can
    collapse ten nominal bins into two, and an ECE over two bins is not comparable with one over
    ten without saying so.
    """

    nominal_count: int
    lower: FloatArray
    upper: FloatArray
    occupancy: IntArray
    mean_confidence: FloatArray
    accuracy: FloatArray

    @property
    def effective_count(self) -> int:
        """Number of bins that hold at least one window."""
        return int(self.occupancy.size)

    @property
    def weight(self) -> FloatArray:
        """Each bin's share of the windows."""
        return self.occupancy / self.occupancy.sum()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for persisting beside the metric it explains."""
        return {
            "nominal_count": self.nominal_count,
            "effective_count": self.effective_count,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "occupancy": self.occupancy.tolist(),
            "mean_confidence": self.mean_confidence.tolist(),
            "accuracy": self.accuracy.tolist(),
        }


def calibration_bins(
    confidence: FloatArray,
    correct: BoolArray,
    *,
    bins: int = DEFAULT_ECE_BINS,
    binning: str = DEFAULT_BINNING,
) -> CalibrationBins:
    """Form the confidence bins, keeping only the occupied ones (D23, D39).

    Boundaries come from the confidences alone, never the labels. A confidence equal to a
    boundary goes to the upper bin, and identical confidences always share one.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    edges = _bin_edges(confidence, bins, binning)
    intervals = len(edges) - 1
    assignment = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, intervals - 1)
    occupied = [index for index in range(intervals) if np.any(assignment == index)]
    return CalibrationBins(
        nominal_count=bins,
        lower=np.asarray([edges[i] for i in occupied], dtype=np.float64),
        upper=np.asarray([edges[i + 1] for i in occupied], dtype=np.float64),
        occupancy=np.asarray([int(np.sum(assignment == i)) for i in occupied], dtype=np.int64),
        mean_confidence=np.asarray(
            [confidence[assignment == i].mean() for i in occupied], dtype=np.float64
        ),
        accuracy=np.asarray([correct[assignment == i].mean() for i in occupied], dtype=np.float64),
    )


def expected_calibration_error(
    confidence: FloatArray,
    correct: BoolArray,
    *,
    bins: int = DEFAULT_ECE_BINS,
    binning: str = DEFAULT_BINNING,
) -> float:
    """Weighted mean absolute gap between confidence and accuracy, over adaptive bins (D23)."""
    formed = calibration_bins(confidence, correct, bins=bins, binning=binning)
    return float(np.sum(formed.weight * np.abs(formed.accuracy - formed.mean_confidence)))


def overconfidence_error(
    confidence: FloatArray,
    correct: BoolArray,
    *,
    bins: int = DEFAULT_ECE_BINS,
    binning: str = DEFAULT_BINNING,
) -> float:
    """Confidence-weighted **positive** gap only — one-sided by design (D25)."""
    formed = calibration_bins(confidence, correct, bins=bins, binning=binning)
    gap = np.maximum(formed.mean_confidence - formed.accuracy, 0.0)
    return float(np.sum(formed.weight * formed.mean_confidence * gap))


def mean_confidence_when_wrong(confidence: FloatArray, correct: BoolArray) -> float | None:
    """Mean confidence over misclassified windows. Descriptive only, and ``None`` with no errors.

    Never compared across arms: three perfectly calibrated models score 0.25, 0.40 and 0.67 on
    this purely from their confidence profiles, so no value of it means "calibrated". 0 would
    read as "perfectly humble when wrong", which is why the empty case is ``None`` (D25).
    """
    errors = ~np.asarray(correct, dtype=bool)
    if not errors.any():
        return None
    return float(np.asarray(confidence, dtype=np.float64)[errors].mean())


def per_subject(
    metric: Callable[..., float],
    *arrays: npt.NDArray[np.number],
    subjects: npt.NDArray[np.number],
    **kwargs: object,
) -> dict[int, float]:
    """Apply ``metric`` within each subject separately — never pooled (D26).

    A pooled number can look respectable while one subject sits at chance.
    """
    subject_ids = np.asarray(subjects)
    return {
        int(subject): metric(
            *(np.asarray(a)[subject_ids == subject] for a in arrays),
            **kwargs,
        )
        for subject in np.unique(subject_ids)
    }


def paired_bootstrap_interval(
    per_subject_differences: FloatArray,
    *,
    seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Percentile bootstrap over subjects, for a paired cross-arm difference (D26).

    Every arm is scored on the identical test subjects, so per-subject differences control the
    shared subject variation — it removes the additive subject-level component, not a
    subject-by-arm interaction. A difference is reportable when this interval excludes zero.
    """
    differences = np.asarray(per_subject_differences, dtype=np.float64)
    if differences.size < 2:
        raise ValueError(f"a paired interval needs at least two subjects, got {differences.size}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)

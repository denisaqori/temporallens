"""F1's epoch selection and horizon rule (D18–D21), as pure functions of validation scores.

These are protocol constants, not tuning knobs, so they live here as module constants and the
harness refuses a config that declares different values. Every rule is a function of the
per-epoch scores alone, which is what makes it testable without training anything:

* **The fold score is subject-equal (D21).** Each of the fold's four validation subjects gets its
  own 18-class macro-F1, and the fold score is their unweighted mean. Pooling their windows
  first would let the subject with the most windows dominate, and the protocol generalizes to a
  new *person*.
* **Selection uses a full trailing window (D18).** For epoch *e* the selection score is the mean
  raw fold score over epochs *e*−9 to *e*, so epochs 1–9 are ineligible. The highest score wins
  and an exact tie goes to the earliest epoch. The smoothed value selects a checkpoint; it is
  never reported as the fold's ``macro_f1``, which comes from that checkpoint's predictions (D20).
* **The common budget is the ceiling of the median (D18).** With eight folds that is
  ``ceil((e(4) + e(5)) / 2)`` over the sorted selected epochs. D48 then reads every fold's
  trajectory at that one budget, which may differ from every fold's own peak.
* **One escalation, then a stop (D19).** A budget equal to the 30-epoch horizon is right-censored:
  all eight folds rerun from scratch at 60 epochs. A budget of 60 stops for an owner decision.

Scores are summed with :func:`math.fsum` so the "exactly equal" in D18's tie rule means equal real
sums, not equal up to the order NumPy happened to add them in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

#: D18's trailing window, and therefore the first eligible epoch.
SELECTION_WINDOW_EPOCHS = 10
#: D18's initial cross-validation horizon and D19's single escalated horizon.
INITIAL_HORIZON_EPOCHS = 30
ESCALATED_HORIZON_EPOCHS = 60
#: D6/D16: eight folds of four validation subjects.
EXPECTED_FOLDS = 8
EXPECTED_VALIDATION_SUBJECTS = 4


def fold_score(
    per_subject_macro_f1: Mapping[int, float],
    *,
    expected_subjects: int = EXPECTED_VALIDATION_SUBJECTS,
) -> float:
    """The raw fold score: the unweighted mean of per-subject macro-F1 (D21)."""
    if len(per_subject_macro_f1) != expected_subjects:
        raise ValueError(
            f"a fold score needs exactly {expected_subjects} validation subjects, got "
            f"{sorted(per_subject_macro_f1)}"
        )
    values = [float(v) for v in per_subject_macro_f1.values()]
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"per-subject macro-F1 must lie in [0, 1], got {values}")
    return math.fsum(values) / len(values)


def _require_curve(raw_scores: Sequence[float], window: int) -> list[float]:
    scores = [float(s) for s in raw_scores]
    if len(scores) < window:
        raise ValueError(
            f"a curve of {len(scores)} epoch(s) has no full {window}-epoch selection window"
        )
    if not all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores):
        raise ValueError("raw fold scores must be finite values in [0, 1]")
    return scores


def smoothed_selection_scores(
    raw_scores: Sequence[float], *, window: int = SELECTION_WINDOW_EPOCHS
) -> list[float | None]:
    """D18's trailing-window means, one per epoch; ``None`` where the window is incomplete.

    Position ``i`` is epoch ``i + 1``, so the first ``window - 1`` entries are ``None``.
    """
    scores = _require_curve(raw_scores, window)
    return [
        None if epoch < window else math.fsum(scores[epoch - window : epoch]) / window
        for epoch in range(1, len(scores) + 1)
    ]


@dataclass(frozen=True)
class EpochSelection:
    """One fold's D18 selection: the chosen one-based epoch and the score that chose it.

    ``smoothed_score`` is a selection diagnostic only (D20); the fold's reportable metrics come
    from the predictions of the checkpoint saved at ``epoch``.
    """

    epoch: int
    smoothed_score: float
    raw_scores: tuple[float, ...]
    smoothed_scores: tuple[float | None, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form for the run log."""
        return {
            "epoch": self.epoch,
            "smoothed_score": self.smoothed_score,
            "raw_scores": list(self.raw_scores),
            "smoothed_scores": list(self.smoothed_scores),
        }


def select_epoch(
    raw_scores: Sequence[float], *, window: int = SELECTION_WINDOW_EPOCHS
) -> EpochSelection:
    """The earliest epoch attaining the highest full-window trailing mean (D18)."""
    smoothed = smoothed_selection_scores(raw_scores, window=window)
    eligible = [(epoch, score) for epoch, score in enumerate(smoothed, 1) if score is not None]
    best = max(score for _, score in eligible)
    epoch = next(epoch for epoch, score in eligible if score == best)
    return EpochSelection(
        epoch=epoch,
        smoothed_score=best,
        raw_scores=tuple(float(s) for s in raw_scores),
        smoothed_scores=tuple(smoothed),
    )


def common_budget(selected_epochs: Sequence[int], *, expected_folds: int = EXPECTED_FOLDS) -> int:
    """D18's refit budget: the ceiling of the median selected epoch."""
    epochs = list(selected_epochs)
    if len(epochs) != expected_folds:
        raise ValueError(f"expected {expected_folds} selected epochs, got {len(epochs)}")
    if not all(isinstance(e, int) and not isinstance(e, bool) and e >= 1 for e in epochs):
        raise ValueError(f"selected epochs must be positive integers, got {epochs}")
    ordered = sorted(epochs)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    # Integer arithmetic: the ceiling of a half-integer is exact, never a float rounding.
    return -(-(ordered[middle - 1] + ordered[middle]) // 2)


@dataclass(frozen=True)
class HorizonDecision:
    """What D19 says to do with a common budget found at a given horizon."""

    action: Literal["accept", "escalate", "block"]
    budget: int
    horizon: int
    next_horizon: int | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe form for the run log."""
        return {
            "action": self.action,
            "budget": self.budget,
            "horizon": self.horizon,
            "next_horizon": self.next_horizon,
        }


def decide_horizon(budget: int, horizon: int) -> HorizonDecision:
    """Accept a budget inside the horizon; escalate 30 to 60 once; block a second boundary hit."""
    if horizon not in (INITIAL_HORIZON_EPOCHS, ESCALATED_HORIZON_EPOCHS):
        raise ValueError(
            f"horizon must be {INITIAL_HORIZON_EPOCHS} or {ESCALATED_HORIZON_EPOCHS} (D18/D19), "
            f"got {horizon}"
        )
    if not SELECTION_WINDOW_EPOCHS <= budget <= horizon:
        raise ValueError(
            f"a budget of {budget} is impossible at horizon {horizon}: selection is confined to "
            f"epochs {SELECTION_WINDOW_EPOCHS}..{horizon}"
        )
    if budget < horizon:
        return HorizonDecision("accept", budget, horizon)
    if horizon == INITIAL_HORIZON_EPOCHS:
        return HorizonDecision("escalate", budget, horizon, ESCALATED_HORIZON_EPOCHS)
    return HorizonDecision("block", budget, horizon)

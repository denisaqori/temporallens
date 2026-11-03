"""Temperature scaling: D33's fitting weights, D39's solver and failure rules, D81's filter.

One scalar ``T`` rescales logits as ``softmax(logits / T)``. It never changes an argmax, so
accuracy, macro-F1 and confusion matrices are untouched; it changes confidence, which is what the
calibration metrics read. Everything here is about fitting that one number honestly:

* **Weights, not window counts (D33).** Every fitting subject weighs the same, every recording
  repetition weighs the same within its subject, and every window the same within its repetition.
  A subject with more windows must not silently dominate the population calibrator.
* **Repetition 0 never enters a fit (D81).** Each DB2 recording ends with one repetition-0 rest
  segment of a few dozen windows (31, 13 and 28 in subjects 1–3). Weighted as a seventh repetition
  it would carry about a seventh of its subject's weight while being pure rest, so it gets weight
  0 here. It still counts in selection scores and evaluation metrics.
* **One recipe, and a fit that fails says so (D39).** Minimize the weighted NLL in
  ``beta = 1/T`` with SciPy's bounded scalar solver in float64, bounds ``[0.01, 100]``,
  ``xatol=1e-8``, ``maxiter=500``. The NLL is convex in ``beta``: its second derivative is the
  weighted variance of the logits under the scaled softmax. A boundary optimum, a solver that did
  not converge, or a fit worse than ``T=1`` by more than ``1e-8`` raises
  :class:`TemperatureFitError`; nothing clips, widens the bounds or falls back to ``T=1``.
* **No fitted temperature scores its own fitting set as validation (D40).**
  :func:`cross_fitted_temperatures` fits on all-but-one subject and scores only the excluded one,
  for each subject in turn. The temperatures it returns are diagnostics; they never feed a refit.

The derivative of the NLL in ``beta`` is ``sum_i w_i (E_p[z_i] - z_i[y_i])``. At the lower bound it
must be negative and at the upper bound positive, or the minimum is not inside the bounds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: D39's predeclared numerical guards; equivalently T in [0.01, 100].
INVERSE_TEMPERATURE_BOUNDS = (0.01, 100.0)
SOLVER = "scipy_minimize_scalar_bounded"
SOLVER_TOLERANCE = 1e-8
SOLVER_MAX_ITERATIONS = 500
NLL_NON_REGRESSION_TOLERANCE = 1e-8
#: D81: the terminal rest segment is excluded from every temperature fit.
EXCLUDED_FITTING_REPETITIONS: tuple[int, ...] = (0,)


class TemperatureFitError(RuntimeError):
    """A temperature fit that D39 says must block reportable scaling rather than be repaired."""


def d33_fitting_weights(
    subjects: npt.ArrayLike,
    repetitions: npt.ArrayLike,
    *,
    excluded_repetitions: Sequence[int] = EXCLUDED_FITTING_REPETITIONS,
) -> FloatArray:
    """Normalized D33 weights: equal subjects, equal repetitions within each, equal windows within.

    Windows in ``excluded_repetitions`` get weight 0 (D81). A subject left with no eligible
    window is an error rather than a silently smaller fitting population.
    """
    subject = np.asarray(subjects, dtype=np.int64)
    repetition = np.asarray(repetitions, dtype=np.int64)
    if subject.ndim != 1 or subject.shape != repetition.shape or subject.size == 0:
        raise ValueError("subjects and repetitions must be non-empty 1-D arrays of equal length")
    eligible = ~np.isin(repetition, np.asarray(list(excluded_repetitions), dtype=np.int64))
    weights = np.zeros(subject.size, dtype=np.float64)
    fitting_subjects = np.unique(subject)
    for one in fitting_subjects:
        own = (subject == one) & eligible
        if not own.any():
            raise ValueError(
                f"subject {one} has no window outside repetitions {list(excluded_repetitions)}, "
                "so it cannot enter the fit"
            )
        kept = np.unique(repetition[own])
        for rep in kept:
            rows = own & (repetition == rep)
            weights[rows] = 1.0 / (fitting_subjects.size * kept.size * rows.sum())
    return weights / math.fsum(weights.tolist())


def _check_fit_inputs(
    logits: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike
) -> tuple[FloatArray, IntArray, FloatArray]:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    w = np.asarray(weights, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise TemperatureFitError("logits must be a non-empty (windows, classes) array")
    if y.ndim != 1 or y.size != z.shape[0] or not np.issubdtype(y.dtype, np.integer):
        raise TemperatureFitError("labels must be one integer per logit row")
    if y.min() < 0 or y.max() >= z.shape[1]:
        raise TemperatureFitError("labels must index the logit columns")
    if w.shape != y.shape:
        raise TemperatureFitError("weights must be one value per logit row")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(w))):
        raise TemperatureFitError("logits and weights must be finite")
    if w.min() < 0 or not math.isclose(math.fsum(w.tolist()), 1.0, abs_tol=1e-12):
        raise TemperatureFitError("weights must be non-negative and normalized to sum to 1")
    return z, y.astype(np.int64), w


def weighted_nll(
    logits: FloatArray, labels: IntArray, weights: FloatArray, inverse_temperature: float
) -> float:
    """Weighted mean NLL of ``softmax(beta * logits)``, by stable log-sum-exp in float64."""
    scaled = inverse_temperature * logits
    per_window = logsumexp(scaled, axis=1) - scaled[np.arange(labels.size), labels]
    return float(np.dot(weights, per_window))


def weighted_nll_derivative(
    logits: FloatArray, labels: IntArray, weights: FloatArray, inverse_temperature: float
) -> float:
    """``d NLL / d beta``: the weighted mean of the expected minus the true-class logit."""
    probabilities = softmax(inverse_temperature * logits, axis=1)
    expected = np.sum(probabilities * logits, axis=1)
    return float(np.dot(weights, expected - logits[np.arange(labels.size), labels]))


@dataclass(frozen=True)
class TemperatureFit:
    """A fitted temperature and every diagnostic D39 requires to be persisted with it."""

    temperature: float
    inverse_temperature: float
    nll: float
    nll_at_temperature_one: float
    lower_bound_derivative: float
    upper_bound_derivative: float
    iterations: int
    converged: bool
    window_count: int
    weighted_window_count: int
    solver: str = SOLVER
    parameter: str = "inverse_temperature"
    bounds: tuple[float, float] = INVERSE_TEMPERATURE_BOUNDS
    tolerance: float = SOLVER_TOLERANCE
    max_iterations: int = SOLVER_MAX_ITERATIONS

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for ``model_config.probability_calibration`` and the run log."""
        out = asdict(self)
        out["bounds"] = list(self.bounds)
        return out


def fit_temperature(
    logits: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike
) -> TemperatureFit:
    """Fit D39's temperature, raising :class:`TemperatureFitError` on any failure it names."""
    z, y, w = _check_fit_inputs(logits, labels, weights)
    lower, upper = INVERSE_TEMPERATURE_BOUNDS

    lower_derivative = weighted_nll_derivative(z, y, w, lower)
    upper_derivative = weighted_nll_derivative(z, y, w, upper)
    if not (math.isfinite(lower_derivative) and math.isfinite(upper_derivative)):
        raise TemperatureFitError("the NLL derivative is not finite at the bounds")
    # Boundary checks come first: they also catch a flat, unidentified objective.
    if lower_derivative >= 0:
        raise TemperatureFitError(
            f"NLL derivative at beta={lower} is {lower_derivative:.3g} >= 0, so the optimum is at "
            "or below the lower bound; D39 blocks rather than widening the range"
        )
    if upper_derivative <= 0:
        raise TemperatureFitError(
            f"NLL derivative at beta={upper} is {upper_derivative:.3g} <= 0, so the optimum is at "
            "or above the upper bound; D39 blocks rather than widening the range"
        )

    result = minimize_scalar(
        lambda beta: weighted_nll(z, y, w, float(beta)),
        bounds=INVERSE_TEMPERATURE_BOUNDS,
        method="bounded",
        options={"xatol": SOLVER_TOLERANCE, "maxiter": SOLVER_MAX_ITERATIONS},
    )
    beta = float(result.x)
    if not result.success or not math.isfinite(beta) or not lower < beta < upper:
        raise TemperatureFitError(
            f"the bounded solver did not converge to an interior optimum: {result.message}"
        )
    fitted = weighted_nll(z, y, w, beta)
    baseline = weighted_nll(z, y, w, 1.0)
    if not math.isfinite(fitted) or fitted > baseline + NLL_NON_REGRESSION_TOLERANCE:
        raise TemperatureFitError(
            f"the fitted NLL {fitted:.12g} is worse than T=1 ({baseline:.12g}) by more than "
            f"{NLL_NON_REGRESSION_TOLERANCE}"
        )
    return TemperatureFit(
        temperature=1.0 / beta,
        inverse_temperature=beta,
        nll=fitted,
        nll_at_temperature_one=baseline,
        lower_bound_derivative=lower_derivative,
        upper_bound_derivative=upper_derivative,
        iterations=int(result.nit),
        converged=bool(result.success),
        window_count=int(y.size),
        weighted_window_count=int(np.count_nonzero(w)),
    )


def scaled_probabilities(logits: npt.ArrayLike, temperature: float) -> FloatArray:
    """``softmax(logits / T)`` in float64, for a finite positive ``T``."""
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
    z = np.asarray(logits, dtype=np.float64)
    return np.asarray(softmax(z / temperature, axis=1), dtype=np.float64)


def fit_population_temperature(
    logits: npt.ArrayLike,
    labels: npt.ArrayLike,
    subjects: npt.ArrayLike,
    repetitions: npt.ArrayLike,
) -> TemperatureFit:
    """One D33/D39 fit over pooled windows, weighted by subject and repetition (D81 applied)."""
    return fit_temperature(logits, labels, d33_fitting_weights(subjects, repetitions))


def cross_fitted_temperatures(
    logits: npt.ArrayLike,
    labels: npt.ArrayLike,
    subjects: npt.ArrayLike,
    repetitions: npt.ArrayLike,
) -> dict[int, TemperatureFit]:
    """D40: for each subject, the temperature fitted on every *other* subject.

    The returned temperature for subject ``s`` is the one that may score ``s``; ``s`` contributed
    nothing to it. Use the same frozen model's logits for every fit.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    subject = np.asarray(subjects, dtype=np.int64)
    repetition = np.asarray(repetitions, dtype=np.int64)
    present = np.unique(subject)
    if present.size < 2:
        raise ValueError("cross-fitting needs at least two subjects")
    out: dict[int, TemperatureFit] = {}
    for scored in present:
        fitting = subject != scored
        out[int(scored)] = fit_population_temperature(
            z[fitting], y[fitting], subject[fitting], repetition[fitting]
        )
    return out

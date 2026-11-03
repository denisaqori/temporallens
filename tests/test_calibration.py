"""D33's fitting weights, D39's solver and failure rules, D40's cross-fitting and D81's filter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from temporallens.evaluation.calibration import (
    INVERSE_TEMPERATURE_BOUNDS,
    TemperatureFitError,
    cross_fitted_temperatures,
    d33_fitting_weights,
    fit_population_temperature,
    fit_temperature,
    scaled_probabilities,
    weighted_nll,
    weighted_nll_derivative,
)
from temporallens.training.class_weights import (
    CLASS_WEIGHTS_DIGEST_ENCODING,
    class_weights_sha256,
    inverse_frequency_weights,
)


def _sampled(temperature: float, *, windows: int = 4000, classes: int = 5, seed: int = 0):
    """Labels drawn from softmax(logits / T): a model whose true temperature is ``temperature``."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 3.0, (windows, classes))
    probabilities = softmax(logits / temperature, axis=1)
    labels = np.asarray([rng.choice(classes, p=p) for p in probabilities], dtype=np.int64)
    return logits, labels


def test_weights_are_equal_per_subject_then_repetition_then_window() -> None:
    """D33: a subject with more windows or repetitions does not gain weight."""
    subjects = np.asarray([1, 1, 1, 1, 2, 2])
    repetitions = np.asarray([1, 1, 1, 2, 1, 1])
    weights = d33_fitting_weights(subjects, repetitions)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[subjects == 1].sum() == pytest.approx(0.5)
    assert weights[(subjects == 1) & (repetitions == 1)].sum() == pytest.approx(0.25)
    assert weights[3] == pytest.approx(0.25)  # alone in its repetition


def test_repetition_zero_gets_no_fitting_weight() -> None:
    """D81: the terminal rest segment never acts as a seventh repetition."""
    subjects = np.asarray([1, 1, 1, 2, 2])
    repetitions = np.asarray([0, 1, 2, 0, 1])
    weights = d33_fitting_weights(subjects, repetitions)
    assert weights[repetitions == 0].tolist() == [0.0, 0.0]
    assert weights[subjects == 1].sum() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no window outside"):
        d33_fitting_weights(np.asarray([1, 2]), np.asarray([1, 0]))


@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.5])
def test_a_known_temperature_is_recovered(true_temperature: float) -> None:
    logits, labels = _sampled(true_temperature)
    weights = np.full(labels.size, 1.0 / labels.size)
    fit = fit_temperature(logits, labels, weights)
    assert fit.temperature == pytest.approx(true_temperature, rel=0.1)
    assert fit.converged and fit.nll <= fit.nll_at_temperature_one + 1e-8
    assert fit.lower_bound_derivative < 0 < fit.upper_bound_derivative


def test_the_derivative_matches_a_finite_difference() -> None:
    logits, labels = _sampled(1.5, windows=300)
    weights = np.full(labels.size, 1.0 / labels.size)
    step = 1e-6
    numeric = (
        weighted_nll(logits, labels, weights, 0.7 + step)
        - weighted_nll(logits, labels, weights, 0.7 - step)
    ) / (2 * step)
    assert weighted_nll_derivative(logits, labels, weights, 0.7) == pytest.approx(numeric, rel=1e-5)


def test_a_boundary_optimum_blocks_instead_of_clipping() -> None:
    """Every window confidently right: NLL keeps falling past the upper bound, so D39 blocks."""
    logits = np.zeros((50, 3))
    labels = np.arange(50) % 3
    logits[np.arange(50), labels] = 5.0
    with pytest.raises(TemperatureFitError, match="upper bound"):
        fit_temperature(logits, labels, np.full(50, 1 / 50))


def test_a_flat_objective_blocks() -> None:
    """Identical logits carry no confidence to rescale: the fit is unidentified."""
    with pytest.raises(TemperatureFitError):
        fit_temperature(np.zeros((10, 3)), np.arange(10) % 3, np.full(10, 0.1))


def test_unnormalized_or_non_finite_inputs_are_refused() -> None:
    logits, labels = _sampled(1.0, windows=20)
    with pytest.raises(TemperatureFitError, match="normalized"):
        fit_temperature(logits, labels, np.ones(20))
    logits[0, 0] = np.inf
    with pytest.raises(TemperatureFitError, match="finite"):
        fit_temperature(logits, labels, np.full(20, 0.05))


def test_bounds_are_d39s() -> None:
    assert INVERSE_TEMPERATURE_BOUNDS == (0.01, 100.0)


def test_scaling_never_changes_a_prediction() -> None:
    logits, _ = _sampled(1.0, windows=200)
    assert np.array_equal(scaled_probabilities(logits, 3.0).argmax(axis=1), logits.argmax(axis=1))
    with pytest.raises(ValueError, match="positive"):
        scaled_probabilities(logits, 0.0)


def test_cross_fitting_never_lets_a_subject_into_its_own_calibrator() -> None:
    """D40: subject s is scored with a T fitted on the others; corrupting s cannot move it."""
    logits, labels = _sampled(1.5, windows=1200)
    subjects = np.repeat([1, 2, 3, 4], 300)
    repetitions = np.tile(np.repeat([1, 2, 3], 100), 4)
    fits = cross_fitted_temperatures(logits, labels, subjects, repetitions)
    assert sorted(fits) == [1, 2, 3, 4]

    shuffled = labels.copy()
    shuffled[subjects == 1] = np.random.default_rng(1).permutation(shuffled[subjects == 1])
    refitted = cross_fitted_temperatures(logits, shuffled, subjects, repetitions)
    assert refitted[1].temperature == pytest.approx(fits[1].temperature, rel=1e-6)
    assert refitted[2].temperature != pytest.approx(fits[2].temperature, rel=1e-6)


def test_population_fit_uses_d33_weights() -> None:
    """Duplicating one subject's windows changes nothing: its weight stays one subject's."""
    logits, labels = _sampled(1.5, windows=600)
    subjects = np.repeat([1, 2], 300)
    repetitions = np.tile(np.repeat([1, 2], 150), 2)
    base = fit_population_temperature(logits, labels, subjects, repetitions)
    one = subjects == 1
    doubled = fit_population_temperature(
        np.concatenate([logits, logits[one]]),
        np.concatenate([labels, labels[one]]),
        np.concatenate([subjects, subjects[one]]),
        np.concatenate([repetitions, repetitions[one]]),
    )
    assert doubled.temperature == pytest.approx(base.temperature, rel=1e-6)


def test_the_class_weight_digest_is_stable_and_detects_any_change() -> None:
    """D85: producer and consumer hash the same float32 bytes under a versioned header."""
    weights = inverse_frequency_weights(np.asarray([0, 0, 0, 1, 2, 2]), num_classes=18)
    digest = class_weights_sha256(weights)
    assert digest == class_weights_sha256(weights.copy())
    assert len(digest) == 64
    changed = weights.copy()
    changed[1] = np.nextafter(changed[1], np.float32(10))
    assert class_weights_sha256(changed) != digest
    assert CLASS_WEIGHTS_DIGEST_ENCODING == "d11_class_weights_v1_shape_dtype_little_endian_float32"


def test_the_class_weight_digest_refuses_a_vector_no_trainer_used() -> None:
    with pytest.raises(ValueError, match="float32"):
        class_weights_sha256(np.ones(18, dtype=np.float64))
    with pytest.raises(ValueError, match="1-D"):
        class_weights_sha256(np.ones((2, 9), dtype=np.float32))

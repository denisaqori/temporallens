"""Per-channel normalization: what the statistics are fitted on, and that they cannot leak.

Section 3.3 calls normalization the most common silent leak in the project, so most of these
tests are about the *fitting set* rather than the arithmetic.

DESIGN DECISION UNDER TEST (needs owner review): statistics are fitted over the samples that
training windows actually COVER, each counted once -- not over windows, and not over whole
recordings. See test_covered_samples_are_counted_once and
test_samples_no_window_can_reach_are_excluded for why each alternative is worse.
"""

from __future__ import annotations

import numpy as np
import pytest
from _synthetic import recording, recordings

from temporallens.data.ninapro import NUM_CHANNELS
from temporallens.data.windows import index_recording
from temporallens.preprocessing.normalize import (
    ChannelStats,
    apply_channel_stats,
    fit_channel_stats,
)


def _index(rec, stride=100, window=400):
    return index_recording(rec, window_size=window, stride=stride)


# --- shape and arithmetic -------------------------------------------------------------------


def test_statistics_are_per_channel() -> None:
    rec = recording(1, seed=1)
    stats = fit_channel_stats({1: rec}, _index(rec), fitted_on="debug")
    assert stats.mean.shape == (NUM_CHANNELS,)
    assert stats.std.shape == (NUM_CHANNELS,)
    # The fixture scales channel c by (1 + c), so the stds must be strictly increasing.
    assert np.all(np.diff(stats.std) > 0), "per-channel stds collapsed into one value"


def test_applying_stats_standardises_the_fitting_set() -> None:
    rec = recording(1, seed=2)
    index = _index(rec)
    stats = fit_channel_stats({1: rec}, index, fitted_on="debug")

    from temporallens.data.windows import materialise

    data = apply_channel_stats(stats, materialise(index, {1: rec}))
    # Per channel over all windows: close to zero mean and unit std. Not exact, because windows
    # oversample interior positions -- which is precisely why the stats are not fitted on them.
    per_channel_mean = data.mean(axis=(0, 2))
    per_channel_std = data.std(axis=(0, 2))
    assert np.abs(per_channel_mean).max() < 0.15
    assert np.abs(per_channel_std - 1.0).max() < 0.15


def test_apply_is_shape_preserving_and_float32() -> None:
    rec = recording(1, seed=3)
    index = _index(rec)
    stats = fit_channel_stats({1: rec}, index, fitted_on="debug")

    from temporallens.data.windows import materialise

    raw = materialise(index, {1: rec})
    out = apply_channel_stats(stats, raw)
    assert out.shape == raw.shape
    assert out.dtype == np.float32


# --- the fitting-set decision ---------------------------------------------------------------


def test_covered_samples_are_counted_once() -> None:
    """Fitting over windows would weight an interior sample four times at stride 100.

    A sample in the middle of a long segment belongs to four windows; one near a segment edge
    belongs to fewer. Fitting over windows therefore pulls the mean toward segment interiors.
    Counting each covered sample once removes that weighting.
    """
    rec = recording(1, seed=4)
    index = _index(rec, stride=100)
    stats = fit_channel_stats({1: rec}, index, fitted_on="debug")

    from temporallens.data.windows import materialise

    window_mean = materialise(index, {1: rec}).mean(axis=(0, 2))
    # The two differ: if they were identical, the implementation is averaging windows.
    assert not np.allclose(
        stats.mean, window_mean, atol=1e-7
    ), "statistics match the window mean, so overlapping samples are being double-counted"


def test_samples_no_window_can_reach_are_excluded() -> None:
    """Fitting over the whole recording would include the tail of every segment.

    D24 drops the tail of each segment where fewer than window_size samples remain, so those
    samples are never shown to the model. Statistics describing them describe a distribution
    that does not exist downstream.
    """
    # Segment length matters: coverage is 100% whenever (L - window) is divisible by stride,
    # which 1000/400/100 happens to be. 1037 leaves a 37-sample tail per segment, so the test
    # has something to detect. Asserting the premise keeps it from going vacuous if the fixture
    # is ever changed back to a round number.
    rec = recording(1, gestures=1, repetitions=1, samples_per_segment=1037, seed=5)
    index = _index(rec, stride=100, window=400)
    covered = np.zeros(rec.num_samples, dtype=bool)
    for start in index.start:
        covered[start : start + index.window_size] = True
    assert covered.sum() < rec.num_samples, "premise failed: every sample is covered"

    stats = fit_channel_stats({1: rec}, index, fitted_on="debug")
    whole_recording_mean = rec.emg.mean(axis=0)
    assert not np.allclose(
        stats.mean, whole_recording_mean, atol=1e-7
    ), "statistics match the whole-recording mean, so uncovered segment tails are included"


def test_coverage_is_the_union_of_window_spans() -> None:
    """With stride == window_size there is no overlap, so covered-once equals the window mean."""
    rec = recording(1, seed=6)
    index = _index(rec, stride=400, window=400)
    stats = fit_channel_stats({1: rec}, index, fitted_on="debug")

    from temporallens.data.windows import materialise

    window_mean = materialise(index, {1: rec}).mean(axis=(0, 2))
    np.testing.assert_allclose(stats.mean, window_mean, rtol=1e-5, atol=1e-6)


# --- leakage ---------------------------------------------------------------------------------


def test_only_the_supplied_subjects_contribute() -> None:
    """The fitting set is an explicit argument; there is no path that sees every subject."""
    recs = recordings((1, 2, 3))
    train_index = index_recording(recs[1], window_size=400, stride=200)

    fitted = fit_channel_stats({1: recs[1]}, train_index, fitted_on="fold")
    all_subjects = fit_channel_stats(
        recs,
        __import__("temporallens.data.windows", fromlist=["concatenate"]).concatenate(
            [index_recording(r, window_size=400, stride=200) for r in recs.values()]
        ),
        fitted_on="all",
    )
    assert not np.allclose(fitted.mean, all_subjects.mean), (
        "statistics did not change when two more subjects were added, so the fitting set is "
        "not actually being respected"
    )


def test_fitting_an_index_without_its_recording_is_an_error() -> None:
    rec = recording(1, seed=7)
    with pytest.raises(KeyError, match="no recording supplied"):
        fit_channel_stats({}, _index(rec), fitted_on="debug")


def test_provenance_is_recorded() -> None:
    """`fitted_on` exists so a checkpoint can say which partition produced its statistics."""
    rec = recording(1, seed=8)
    stats = fit_channel_stats({1: rec}, _index(rec), fitted_on="fold_3_training_28")
    assert stats.fitted_on == "fold_3_training_28"
    assert stats.num_samples > 0


# --- degenerate channels ---------------------------------------------------------------------


def test_a_dead_channel_does_not_divide_by_zero() -> None:
    """A flat channel has zero variance. EMG channels do fail, so this must not produce inf."""
    rec = recording(1, seed=9)
    emg = rec.emg.copy()
    emg[:, 3] = 0.0
    from temporallens.data.ninapro import SubjectRecording

    flat = SubjectRecording(1, emg, rec.label, rec.repetition)
    index = _index(flat)
    stats = fit_channel_stats({1: flat}, index, fitted_on="debug")

    assert stats.std[3] > 0.0, "a zero-variance channel must be floored, not left at 0"

    from temporallens.data.windows import materialise

    out = apply_channel_stats(stats, materialise(index, {1: flat}))
    assert np.isfinite(out).all(), "normalizing a dead channel produced non-finite values"


# --- serialisation into the checkpoint -------------------------------------------------------


def test_round_trip_through_a_plain_dict() -> None:
    """Statistics travel inside `model_config`, so they must serialise without numpy types.

    DESIGN DECISION UNDER TEST (needs owner review): the encoder is frozen and reused by both
    arms, so if its statistics do not travel with the checkpoint the arms normalize differently
    and the F1 reference row stops being comparable. Same argument as D23's temperature.
    """
    rec = recording(1, seed=10)
    stats = fit_channel_stats({1: rec}, _index(rec), fitted_on="debug")

    payload = stats.to_dict()
    import json

    assert json.loads(json.dumps(payload)) == payload, "statistics are not JSON-serialisable"

    restored = ChannelStats.from_dict(payload)
    np.testing.assert_allclose(restored.mean, stats.mean)
    np.testing.assert_allclose(restored.std, stats.std)
    assert restored.fitted_on == stats.fitted_on

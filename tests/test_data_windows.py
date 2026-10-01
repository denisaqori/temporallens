"""Loading and windowing contracts for NinaPro DB2 Exercise B.

The fixtures here are synthetic recordings with DB2's *shape*, not DB2 data. They exist only to
exercise this code, they are built in ``tmp_path``, and nothing computed from them is reportable.
The real dataset is downloaded under its own terms and is never committed or fabricated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from temporallens.data.ninapro import (
    NUM_CHANNELS,
    NUM_CLASSES,
    NinaProFormatError,
    SubjectRecording,
    available_subjects,
    load_processed,
    processed_path,
    read_raw_mat,
    save_processed,
)
from temporallens.data.windows import (
    concatenate,
    index_recording,
    materialise,
    segment_spans,
)

SAMPLES_PER_SEGMENT = 1000


def _synthetic_recording(
    subject: int = 1,
    *,
    gestures: int = 3,
    repetitions: int = 2,
    samples_per_segment: int = SAMPLES_PER_SEGMENT,
) -> SubjectRecording:
    """A rest/gesture/rest/gesture… recording shaped like Exercise B.

    Each channel carries the segment's label as a constant offset, so a window's content is
    checkable against the label it was given.
    """
    labels: list[int] = []
    reps: list[int] = []
    for repetition in range(1, repetitions + 1):
        for gesture in range(1, gestures + 1):
            labels += [0] * samples_per_segment + [gesture] * samples_per_segment
            # Shipped DB2 rest usually carries the adjacent trial's corrected repetition index.
            reps += [repetition] * samples_per_segment * 2

    label = np.asarray(labels, dtype=np.int16)
    repetition = np.asarray(reps, dtype=np.int16)
    emg = np.repeat(label.astype(np.float32)[:, None], NUM_CHANNELS, axis=1)
    return SubjectRecording(subject=subject, emg=emg, label=label, repetition=repetition)


# --- the recording contract ---------------------------------------------------------------


def test_recording_rejects_a_wrong_channel_count() -> None:
    with pytest.raises(NinaProFormatError, match="emg must be"):
        SubjectRecording(
            subject=1,
            emg=np.zeros((100, 8), dtype=np.float32),
            label=np.zeros(100, dtype=np.int16),
            repetition=np.zeros(100, dtype=np.int16),
        )


def test_recording_rejects_labels_outside_exercise_b() -> None:
    """A label above 17 usually means a different exercise file was read."""
    label = np.zeros(100, dtype=np.int16)
    label[0] = NUM_CLASSES  # one past the last valid class
    with pytest.raises(NinaProFormatError, match="Exercise B"):
        SubjectRecording(
            subject=1,
            emg=np.zeros((100, NUM_CHANNELS), dtype=np.float32),
            label=label,
            repetition=np.zeros(100, dtype=np.int16),
        )


def test_recording_rejects_misaligned_streams() -> None:
    with pytest.raises(NinaProFormatError, match="label must be"):
        SubjectRecording(
            subject=1,
            emg=np.zeros((100, NUM_CHANNELS), dtype=np.float32),
            label=np.zeros(99, dtype=np.int16),
            repetition=np.zeros(100, dtype=np.int16),
        )


def _write_raw_mat(
    path,
    *,
    subject: int = 1,
    exercise: int = 1,
    gestures: int = 17,
    corrected: bool = True,
) -> None:
    recording = _synthetic_recording(
        subject=subject,
        gestures=gestures,
        repetitions=6,
        samples_per_segment=10,
    )
    payload = {
        "subject": np.asarray([[subject]], dtype=np.uint8),
        "exercise": np.asarray([[exercise]], dtype=np.uint8),
        "emg": recording.emg,
        "stimulus": recording.label[:, None],
        "repetition": recording.repetition[:, None],
    }
    if corrected:
        payload |= {
            "restimulus": recording.label[:, None],
            "rerepetition": recording.repetition[:, None],
        }
    savemat(path, payload)


def test_raw_mat_identity_matches_the_requested_subject_and_exercise(tmp_path) -> None:
    path = tmp_path / "S7_E1_A1.mat"
    _write_raw_mat(path, subject=7)
    recording = read_raw_mat(path, subject=7)
    assert recording.subject == 7


def test_raw_mat_rejects_a_subject_mapping_mismatch(tmp_path) -> None:
    path = tmp_path / "misnamed.mat"
    _write_raw_mat(path, subject=17)
    with pytest.raises(NinaProFormatError, match="caller requested subject 1"):
        read_raw_mat(path, subject=1)


def test_raw_mat_rejects_a_non_exercise_b_file_even_if_labels_fit(tmp_path) -> None:
    path = tmp_path / "S1_E2_A1.mat"
    _write_raw_mat(path, exercise=2)
    with pytest.raises(NinaProFormatError, match="expected Exercise B"):
        read_raw_mat(path, subject=1)


def test_raw_mat_rejects_incomplete_gesture_repetition_coverage(tmp_path) -> None:
    path = tmp_path / "incomplete.mat"
    _write_raw_mat(path, gestures=16)
    with pytest.raises(NinaProFormatError, match="incomplete Exercise-B"):
        read_raw_mat(path, subject=1)


# --- the processed round trip -------------------------------------------------------------


def test_processed_round_trip_preserves_every_stream(tmp_path) -> None:
    recording = _synthetic_recording(subject=7)
    path = processed_path(tmp_path, 7)
    save_processed(recording, path)
    restored = load_processed(path)

    assert restored.subject == 7
    np.testing.assert_array_equal(restored.emg, recording.emg)
    np.testing.assert_array_equal(restored.label, recording.label)
    np.testing.assert_array_equal(restored.repetition, recording.repetition)


def test_processed_file_built_from_uncorrected_labels_is_rejected(tmp_path) -> None:
    """D12: a file prepared from `stimulus` must not load silently."""
    recording = _synthetic_recording()
    path = processed_path(tmp_path, 1)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int16),
        subject=np.asarray(1, dtype=np.int16),
        emg=recording.emg,
        label=recording.label,
        repetition=recording.repetition,
        label_column=np.asarray("stimulus"),
        repetition_column=np.asarray("repetition"),
    )
    with pytest.raises(NinaProFormatError, match="prepare_dataset"):
        load_processed(path)


def test_processed_file_built_from_uncorrected_repetitions_is_rejected(tmp_path) -> None:
    recording = _synthetic_recording()
    path = processed_path(tmp_path, 1)
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.int16),
        subject=np.asarray(1, dtype=np.int16),
        emg=recording.emg,
        label=recording.label,
        repetition=recording.repetition,
        label_column=np.asarray("restimulus"),
        repetition_column=np.asarray("repetition"),
    )
    with pytest.raises(NinaProFormatError, match="rerepetition"):
        load_processed(path)


def test_processed_file_rejects_a_filename_subject_mismatch(tmp_path) -> None:
    recording = _synthetic_recording(subject=2)
    path = processed_path(tmp_path, 1)
    save_processed(recording, path)
    with pytest.raises(NinaProFormatError, match="filename identifies subject 1"):
        load_processed(path)


def test_available_subjects_is_sorted_and_ignores_strays(tmp_path) -> None:
    for subject in (10, 2, 33):
        save_processed(_synthetic_recording(subject), processed_path(tmp_path, subject))
    (tmp_path / "subject_notanumber.npz").write_bytes(b"")
    (tmp_path / "subject_00.npz").write_bytes(b"")
    (tmp_path / "subject_41.npz").write_bytes(b"")
    (tmp_path / "README.txt").write_text("not a recording")

    assert available_subjects(tmp_path) == [2, 10, 33]


def test_available_subjects_on_a_missing_directory_is_empty(tmp_path) -> None:
    assert available_subjects(tmp_path / "nope") == []


# --- segmentation and windowing -----------------------------------------------------------


def test_segments_split_on_both_label_and_repetition_changes() -> None:
    label = np.asarray([0, 0, 1, 1, 1, 0, 0], dtype=np.int16)
    repetition = np.asarray([0, 0, 1, 1, 1, 0, 0], dtype=np.int16)
    assert segment_spans(label, repetition) == [(0, 2), (2, 5), (5, 7)]

    # Same label throughout, but the repetition index advances: still two segments.
    same_label = np.asarray([3, 3, 3, 3], dtype=np.int16)
    two_reps = np.asarray([1, 1, 2, 2], dtype=np.int16)
    assert segment_spans(same_label, two_reps) == [(0, 2), (2, 4)]


def test_no_window_spans_two_segments() -> None:
    """The invariant this module exists for.

    Each channel carries its segment's label as a constant, so a window that straddled a boundary
    would contain two distinct values.
    """
    recording = _synthetic_recording()
    index = index_recording(recording, window_size=400, stride=100)
    data = materialise(index, {recording.subject: recording})

    assert len(index) > 0
    for values, label in zip(data, index.label, strict=True):
        assert np.unique(values).size == 1, "window contains samples from two segments"
        assert float(np.unique(values)[0]) == float(label)


def test_window_labels_and_repetitions_match_their_source_samples() -> None:
    recording = _synthetic_recording()
    windows = index_recording(recording, window_size=400, stride=100)

    for start, label, repetition in zip(
        windows.start, windows.label, windows.repetition, strict=True
    ):
        assert recording.label[start] == label
        assert recording.repetition[start] == repetition


def test_materialised_windows_are_channels_first_for_conv1d() -> None:
    recording = _synthetic_recording()
    index = index_recording(recording, window_size=400, stride=100)
    data = materialise(index, {recording.subject: recording})

    assert data.shape == (len(index), NUM_CHANNELS, 400)
    assert data.dtype == np.float32
    assert index.window_size == 400


def test_stride_controls_window_count_within_a_segment() -> None:
    recording = _synthetic_recording(gestures=1, repetitions=1)
    dense = index_recording(recording, window_size=400, stride=100)
    sparse = index_recording(recording, window_size=400, stride=200)
    assert len(dense) > len(sparse)

    # Two segments of 1000 samples: starts at 0,100,...,600 is 7 windows each at stride 100.
    assert len(dense) == 2 * 7
    assert len(sparse) == 2 * 4


def test_a_segment_shorter_than_one_window_contributes_nothing() -> None:
    recording = _synthetic_recording(gestures=1, repetitions=1, samples_per_segment=100)
    with pytest.raises(ValueError, match="cross a boundary"):
        index_recording(recording, window_size=400, stride=100)


def test_max_windows_subsamples_deterministically_and_keeps_every_present_class() -> None:
    recording = _synthetic_recording()
    first = index_recording(recording, window_size=400, stride=100, max_windows=20, seed=42)
    second = index_recording(recording, window_size=400, stride=100, max_windows=20, seed=42)

    assert len(first) == 20
    np.testing.assert_array_equal(first.start, second.start)
    # A small smoke-test cap must still exercise every class available in its subjects.
    np.testing.assert_array_equal(np.unique(first.label), np.unique(recording.label))


def test_max_windows_above_the_available_count_is_a_no_op() -> None:
    recording = _synthetic_recording()
    everything = index_recording(recording, window_size=400, stride=100)
    capped = index_recording(recording, window_size=400, stride=100, max_windows=10_000)
    assert len(capped) == len(everything)


def test_max_windows_rejects_a_cap_too_small_for_class_coverage() -> None:
    recording = _synthetic_recording()
    with pytest.raises(ValueError, match="cannot retain all"):
        index_recording(recording, window_size=400, stride=100, max_windows=3)


def test_concatenate_preserves_subject_provenance() -> None:
    sets = [
        index_recording(_synthetic_recording(subject), window_size=400, stride=200)
        for subject in (1, 2)
    ]
    joined = concatenate(sets)

    assert len(joined) == sum(len(s) for s in sets)
    assert set(np.unique(joined.subject)) == {1, 2}


def test_concatenate_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty list"):
        concatenate([])


# --- the index/materialise split ------------------------------------------------------------


def test_index_is_cheap_and_holds_no_signal() -> None:
    """The reason indexing and materialising are separate.

    At F1's stride a subject yields ~15,700 windows; materialising a 28-subject fold would need
    gigabytes, because stride-100 windows overlap by 75% and store every sample four times.
    """
    recording = _synthetic_recording()
    index = index_recording(recording, window_size=400, stride=100)

    assert not hasattr(index, "data")
    index_bytes = sum(
        arr.nbytes for arr in (index.subject, index.start, index.label, index.repetition)
    )
    materialised_bytes = materialise(index, {recording.subject: recording}).nbytes
    assert index_bytes * 50 < materialised_bytes


def test_take_selects_a_subset_and_keeps_fields_aligned() -> None:
    recording = _synthetic_recording()
    index = index_recording(recording, window_size=400, stride=100)
    chosen = np.asarray([0, 3, 7], dtype=np.int64)
    subset = index.take(chosen)

    assert len(subset) == 3
    np.testing.assert_array_equal(subset.start, index.start[chosen])
    np.testing.assert_array_equal(subset.label, index.label[chosen])
    assert subset.window_size == index.window_size


def test_for_subjects_filters_by_subject() -> None:
    recordings = {s: _synthetic_recording(s) for s in (1, 2, 3)}
    index = concatenate(
        [index_recording(r, window_size=400, stride=200) for r in recordings.values()]
    )
    subset = index.for_subjects([1, 3])

    assert set(np.unique(subset.subject)) == {1, 3}
    assert len(subset) < len(index)


def test_materialise_spans_multiple_subjects_without_mixing_them() -> None:
    """Rows must come from their own subject's recording, not whichever was iterated first."""
    recordings = {s: _synthetic_recording(s) for s in (1, 2)}
    # Give subject 2 a distinguishable signal: its emg is its label plus 100.
    two = recordings[2]
    recordings[2] = SubjectRecording(
        subject=2, emg=two.emg + 100.0, label=two.label, repetition=two.repetition
    )
    index = concatenate(
        [index_recording(r, window_size=400, stride=200) for r in recordings.values()]
    )
    data = materialise(index, recordings)

    for values, subject, label in zip(data, index.subject, index.label, strict=True):
        expected = float(label) + (100.0 if subject == 2 else 0.0)
        assert float(np.unique(values)[0]) == expected


def test_materialise_without_the_recording_is_an_error() -> None:
    recording = _synthetic_recording(1)
    index = index_recording(recording, window_size=400, stride=200)
    with pytest.raises(KeyError, match="no recording supplied"):
        materialise(index, {})


def test_concatenate_rejects_mismatched_window_sizes() -> None:
    recording = _synthetic_recording()
    with pytest.raises(ValueError, match="differing window sizes"):
        concatenate(
            [
                index_recording(recording, window_size=400, stride=200),
                index_recording(recording, window_size=200, stride=200),
            ]
        )


def test_array_backed_records_do_not_pretend_to_support_equality() -> None:
    """Frozen dataclasses over arrays: the generated __eq__/__hash__ raise, so both are off."""
    a, b = _synthetic_recording(), _synthetic_recording()
    assert a != b  # identity comparison, not an ambiguous array truth value
    assert len({a, b}) == 2  # hashable by identity


# --- reading the shipped .mat -----------------------------------------------------------


def test_read_raw_mat_parses_the_shipped_layout(tmp_path) -> None:
    from temporallens.data.ninapro import read_raw_mat

    path = tmp_path / "S1_E1_A1.mat"
    _write_raw_mat(path)
    recording = read_raw_mat(path, subject=1)

    assert recording.emg.shape == (2040, NUM_CHANNELS)
    assert recording.emg.dtype == np.float32
    # The (n, 1) columns are flattened to (n,).
    assert recording.label.shape == (2040,)
    assert sorted(np.unique(recording.label)) == list(range(18))
    assert sorted(np.unique(recording.repetition)) == list(range(1, 7))


def test_read_raw_mat_refuses_a_file_without_corrected_columns(tmp_path) -> None:
    """D12: `stimulus` is what the software prompted, not what the subject did."""
    from temporallens.data.ninapro import read_raw_mat

    path = tmp_path / "uncorrected.mat"
    _write_raw_mat(path, corrected=False)
    with pytest.raises(NinaProFormatError, match="restimulus"):
        read_raw_mat(path, subject=1)

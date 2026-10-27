"""D44/D45 metadata and test-double trials, never fabricated/reportable NinaPro data."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _synthetic import recording

from temporallens.data.assignments import (
    G3_K_GRID,
    build_f2_assignment,
    build_g3_schedule,
    load_assignment,
    save_assignment,
    trial_pool_sha256,
    window_pool_sha256,
)
from temporallens.data.splits import SplitManifest
from temporallens.data.windows import WindowIndex


def _windows(count: int = 32) -> WindowIndex:
    """An index-only test double; no EMG signal or training result exists."""
    subjects = SplitManifest.load().training_subjects
    return WindowIndex(
        np.asarray([subjects[i % 32] for i in range(count)], dtype=np.int16),
        np.asarray([i // 32 * 100 for i in range(count)], dtype=np.int64),
        np.ones(count, dtype=np.int64),
        np.ones(count, dtype=np.int16),
        400,
    )


def _refresh_envelope(artifact: dict) -> None:
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    artifact["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize("count", [32, 33, 36, 42])
def test_f2_exact_rounding_partition_and_fold_sizes(count: int) -> None:
    artifact = build_f2_assignment(_windows(count), stride=100)
    selected = artifact["assignment"]
    folds = np.asarray(selected["inner_fold"])
    n_test = (count + 4) // 5
    q, r = divmod(count - n_test, 8)
    assert selected["outer_test_count"] == n_test == np.count_nonzero(folds == -1)
    assert selected["inner_fold_counts"] == [q + (i < r) for i in range(8)]
    for fold in range(8):
        train = set(np.flatnonzero((folds >= 0) & (folds != fold)))
        validation = set(np.flatnonzero(folds == fold))
        test = set(np.flatnonzero(folds == -1))
        assert not train & validation and not train & test and not validation & test
        assert train | validation | test == set(range(count))
    assert {r[0] for r in selected["windows"]} == set(SplitManifest.load().training_subjects)


def test_f2_golden_membership_locked_numpy_2_4_6() -> None:
    assert build_f2_assignment(_windows(), stride=100)["assignment"]["inner_fold"] == [
        3,
        4,
        6,
        5,
        6,
        -1,
        0,
        -1,
        -1,
        6,
        1,
        5,
        2,
        0,
        3,
        3,
        -1,
        -1,
        0,
        7,
        -1,
        1,
        7,
        -1,
        7,
        2,
        5,
        0,
        1,
        4,
        4,
        2,
    ]


def test_f2_input_order_does_not_change_membership_or_provenance() -> None:
    index = _windows(42)
    shuffled = index.take(np.random.default_rng(7).permutation(len(index)))
    assert build_f2_assignment(index, stride=100) == build_f2_assignment(shuffled, stride=100)


def test_f2_ignores_labels_and_repetitions_when_assigning() -> None:
    index = _windows(42)
    changed = replace(
        index,
        label=np.full(len(index), 2, dtype=np.int64),
        repetition=np.full(len(index), 4, dtype=np.int16),
    )
    original, alternate = [build_f2_assignment(x, stride=100) for x in (index, changed)]
    assert original["rng"] == alternate["rng"]
    assert original["assignment"]["inner_fold"] == alternate["assignment"]["inner_fold"]
    assert original["assignment_sha256"] == alternate["assignment_sha256"]
    assert original["source_pool_sha256"] != alternate["source_pool_sha256"]


def test_f2_rejects_wrong_subjects_duplicates_shapes_types_and_windowing() -> None:
    index = _windows()
    invalid = [
        index.take(np.arange(31)),
        replace(index, subject=np.full(32, 5, dtype=np.int16)),
        _windows(33).take(np.asarray([*range(32), 0])),
        replace(index, start=np.zeros(31, dtype=np.int64)),
        replace(index, start=np.zeros(32, dtype=float)),
        replace(index, window_size=200),
        replace(index, repetition=np.zeros(32, dtype=np.int16)),
    ]
    for bad in invalid:
        with pytest.raises(ValueError):
            build_f2_assignment(bad, stride=100)
    with pytest.raises(ValueError, match="stride"):
        build_f2_assignment(index, stride=200)
    # Corrected rest can have repetition zero; no label-aware redrawing is permitted.
    rest = replace(
        index, label=np.zeros(32, dtype=np.int64), repetition=np.zeros(32, dtype=np.int16)
    )
    assert build_f2_assignment(rest, stride=100)["assignment"]["inner_fold"] == (
        build_f2_assignment(index, stride=100)["assignment"]["inner_fold"]
    )


@pytest.mark.parametrize("subject", [5, 10])
@pytest.mark.parametrize("schedule", range(5))
def test_g3_all_prefixes_cover_classes_and_balance_repetitions(subject: int, schedule: int) -> None:
    source = recording(subject, gestures=17, repetitions=6, samples_per_segment=400)
    artifact = build_g3_schedule(source, schedule)
    trials = artifact["assignment"]["trials"]
    assert artifact["assignment"]["k_grid"] == list(G3_K_GRID)
    assert sorted(g for g, _ in trials[:17]) == list(range(1, 18))
    assert sorted(trials) == [[g, r] for g in range(1, 18) for r in (1, 4)]
    assert trials[0][1] == (1 if (subject + schedule) % 2 == 0 else 4)
    for k in range(35):
        counts = [sum(r == rep for _, r in trials[:k]) for rep in (1, 4)]
        assert abs(counts[0] - counts[1]) <= 1
    assert [sum(r == rep for _, r in trials[:20]) for rep in (1, 4)] == [10, 10]


def test_g3_golden_three_permutation_call_order_locked_numpy_2_4_6() -> None:
    source = recording(5, gestures=17, repetitions=6, samples_per_segment=400)
    artifact = build_g3_schedule(source, 0)
    assert artifact["rng"]["schedule"]["derived_seed"] == 2693202415252833913
    assert artifact["assignment"]["trials"] == [
        [11, 4],
        [2, 1],
        [6, 4],
        [1, 1],
        [5, 4],
        [17, 1],
        [7, 4],
        [15, 1],
        [10, 4],
        [9, 1],
        [16, 4],
        [13, 1],
        [8, 4],
        [3, 1],
        [12, 4],
        [14, 1],
        [4, 4],
        [6, 1],
        [1, 4],
        [16, 1],
        [3, 4],
        [5, 1],
        [15, 4],
        [10, 1],
        [2, 4],
        [8, 1],
        [14, 4],
        [12, 1],
        [13, 4],
        [7, 1],
        [9, 4],
        [4, 1],
        [17, 4],
        [11, 1],
    ]


def test_g3_source_signal_changes_identity_but_not_trials_or_seed() -> None:
    source = recording(5, gestures=17, repetitions=6, samples_per_segment=400)
    before = build_g3_schedule(source, 0)
    source.emg[(source.label == 1) & (source.repetition == 1)] += 0.5
    after = build_g3_schedule(source, 0)
    assert before["rng"] == after["rng"]
    assert before["assignment_sha256"] == after["assignment_sha256"]
    assert before["source_pool_sha256"] != after["source_pool_sha256"]
    source.emg[source.repetition == 2] += 1
    assert trial_pool_sha256(source) == after["source_pool_sha256"]


def test_g3_refuses_missing_split_or_unwindowable_real_trial_slots() -> None:
    source = recording(5, gestures=17, repetitions=6, samples_per_segment=400)
    active = np.flatnonzero((source.label == 1) & (source.repetition == 1))
    source.label[active[len(active) // 2]] = 0
    with pytest.raises(ValueError, match="complete windowable segment"):
        build_g3_schedule(source, 0)
    short = recording(5, gestures=17, repetitions=6, samples_per_segment=399)
    with pytest.raises(ValueError, match="complete windowable segment"):
        build_g3_schedule(short, 0)


@pytest.mark.parametrize("schedule", [-1, 5, True, 1.5])
def test_g3_refuses_invalid_schedule_indices(schedule: int) -> None:
    source = recording(5, gestures=17, repetitions=6, samples_per_segment=400)
    with pytest.raises(ValueError, match="schedule_index"):
        build_g3_schedule(source, schedule)


@pytest.mark.parametrize("kind", ["f2", "g3"])
def test_artifact_round_trip_exclusive_creation_and_current_source_binding(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = recording(5, gestures=17, repetitions=6, samples_per_segment=400)
    artifact = (
        build_f2_assignment(_windows(42), stride=100)
        if kind == "f2"
        else build_g3_schedule(source, 0)
    )
    expected = (
        window_pool_sha256(_windows(42), stride=100) if kind == "f2" else trial_pool_sha256(source)
    )
    path = tmp_path / "nested" / "assignment.json"
    digest = save_assignment(path, artifact)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        save_assignment(path, artifact)
    assert not list(tmp_path.rglob(".assignment-*"))

    def no_redraw(*args: object, **kwargs: object) -> None:
        raise AssertionError("loading a material assignment must never redraw it")

    monkeypatch.setattr("temporallens.data.assignments.assignment_rng", no_redraw)
    identity = {"expected_schedule_index": 0} if kind == "g3" else {}
    assert (
        load_assignment(
            path, expected_source_pool_sha256=expected, expected_file_sha256=digest, **identity
        )
        == artifact
    )
    with pytest.raises(ValueError, match="current source pool"):
        load_assignment(
            path, expected_source_pool_sha256="0" * 64, expected_file_sha256=digest, **identity
        )


def test_loader_accepts_original_numpy_version_without_changing_assignment(tmp_path: Path) -> None:
    artifact = build_f2_assignment(_windows(), stride=100)
    artifact["rng"]["outer"]["numpy_version"] = "older-library-version-test-double"
    _refresh_envelope(artifact)
    path = tmp_path / "older.json"
    digest = save_assignment(path, artifact)
    assert (
        load_assignment(
            path,
            expected_source_pool_sha256=artifact["source_pool_sha256"],
            expected_file_sha256=digest,
        )
        == artifact
    )


def test_corrupt_checksums_invalid_membership_and_duplicate_json_keys_are_refused(
    tmp_path: Path,
) -> None:
    original = build_f2_assignment(_windows(), stride=100)
    broken = copy.deepcopy(original)
    broken["assignment"]["inner_fold"][0] = 9
    with pytest.raises(ValueError, match="checksum"):
        save_assignment(tmp_path / "bad.json", broken)
    _refresh_envelope(broken)
    with pytest.raises(ValueError, match="fold assignment"):
        save_assignment(tmp_path / "bad.json", broken)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        load_assignment(
            path,
            expected_source_pool_sha256=original["source_pool_sha256"],
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_g3_loader_refuses_another_schedule_from_the_same_source_pool(tmp_path: Path) -> None:
    source = recording(5, gestures=17, repetitions=6, samples_per_segment=400)
    artifact = build_g3_schedule(source, 1)
    path = tmp_path / "schedule1.json"
    digest = save_assignment(path, artifact)
    with pytest.raises(ValueError, match="schedule identity"):
        load_assignment(
            path,
            expected_source_pool_sha256=trial_pool_sha256(source),
            expected_file_sha256=digest,
            expected_schedule_index=0,
        )


def test_trusted_file_digest_refuses_self_consistently_rewritten_membership(tmp_path: Path) -> None:
    artifact = build_f2_assignment(_windows(), stride=100)
    path = tmp_path / "assignment.json"
    trusted = save_assignment(path, artifact)
    folds = artifact["assignment"]["inner_fold"]
    folds[0], folds[1] = folds[1], folds[0]
    material = {
        "window_ids": [r[:2] for r in artifact["assignment"]["windows"]],
        "inner_fold": folds,
    }
    artifact["assignment_sha256"] = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _refresh_envelope(artifact)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="trusted artifact"):
        load_assignment(
            path,
            expected_source_pool_sha256=artifact["source_pool_sha256"],
            expected_file_sha256=trusted,
        )

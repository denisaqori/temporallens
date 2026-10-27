"""D44/D45 metadata and test-double trials, never fabricated/reportable NinaPro data."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
from _synthetic import recording, write_development_recordings

from temporallens.data.assignments import (
    G3_K_GRID,
    build_f2_assignment,
    build_g3_schedule,
    f2_fold_membership,
    f2_source_pool_sha256,
    g3_trial_order,
    load_assignment,
    save_assignment,
    trial_pool_sha256,
)
from temporallens.data.ninapro import (
    NinaProFormatError,
    load_processed,
    processed_path,
    save_processed,
)
from temporallens.data.splits import SplitManifest
from temporallens.data.windows import concatenate, index_recording

# The values D44/D45 produced when they were approved. Any change here changes the protocol.
F2_GOLDEN_32 = [3, 4, 6, 5, 6, -1, 0, -1, -1, 6, 1, 5, 2, 0, 3, 3, -1, -1, 0, 7, -1, 1, 7, -1]
F2_GOLDEN_32 += [7, 2, 5, 0, 1, 4, 4, 2]
G3_GOLDEN_SUBJECT5_SCHEDULE0 = [
    [11, 4], [2, 1], [6, 4], [1, 1], [5, 4], [17, 1], [7, 4], [15, 1], [10, 4], [9, 1],
    [16, 4], [13, 1], [8, 4], [3, 1], [12, 4], [14, 1], [4, 4], [6, 1], [1, 4], [16, 1],
    [3, 4], [5, 1], [15, 4], [10, 1], [2, 4], [8, 1], [14, 4], [12, 1], [13, 4], [7, 1],
    [9, 4], [4, 1], [17, 4], [11, 1],
]  # fmt: skip


@pytest.fixture(scope="module")
def f2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_development_recordings(tmp_path_factory.mktemp("f2_processed"))


@pytest.fixture
def f2_copy(f2_dir: Path, tmp_path: Path) -> Path:
    """A private copy for tests that change a recording."""
    return Path(shutil.copytree(f2_dir, tmp_path / "processed"))


def _g3_source(subject: int = 5):
    return recording(subject, gestures=17, repetitions=6, samples_per_segment=400)


def _refresh(artifact: dict, *, material: bool = False, source: bool = False) -> None:
    """Recompute the self-describing digests, as a careless or malicious rewrite would."""

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
            ).encode()
        ).hexdigest()

    assignment = artifact["assignment"]
    if material:
        artifact["assignment_sha256"] = digest(
            {
                "window_ids": [r[:2] for r in assignment["windows"]],
                "inner_fold": assignment["inner_fold"],
            }
            if artifact["kind"] == "f2_window_assignment_v1"
            else {"subject_id": assignment["subject_id"], "trials": assignment["trials"]}
        )
    if source:
        pool = assignment.get("windows", assignment.get("source_trials"))
        artifact["source_pool_sha256"] = digest(
            {"identity": artifact["source_pool_identity"], "ids": pool}
        )
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    artifact["artifact_sha256"] = digest(payload)


def _rewrite_recording(directory: Path, subject: int, change) -> None:
    path = processed_path(directory, subject)
    original = load_processed(path)
    path.unlink()
    save_processed(change(original), path)


@pytest.mark.parametrize("count", [32, 33, 36, 42])
def test_f2_exact_rounding_partition_and_fold_sizes(count: int) -> None:
    folds = np.asarray(f2_fold_membership(count))
    n_test = (count + 4) // 5
    q, r = divmod(count - n_test, 8)
    assert np.count_nonzero(folds == -1) == n_test
    assert [np.count_nonzero(folds == i) for i in range(8)] == [q + (i < r) for i in range(8)]
    for fold in range(8):
        train = set(np.flatnonzero((folds >= 0) & (folds != fold)))
        validation = set(np.flatnonzero(folds == fold))
        test = set(np.flatnonzero(folds == -1))
        assert not train & validation and not train & test and not validation & test
        assert train | validation | test == set(range(count))


def test_f2_golden_membership_locked_numpy_2_4_6() -> None:
    assert f2_fold_membership(32) == F2_GOLDEN_32


def test_f2_builder_indexes_the_manifest_recordings_itself(f2_dir: Path) -> None:
    manifest = SplitManifest.load()
    artifact = build_f2_assignment(f2_dir)
    expected = concatenate(
        [
            index_recording(load_processed(processed_path(f2_dir, s)), window_size=400, stride=100)
            for s in manifest.training_subjects
        ]
    )
    rows = artifact["assignment"]["windows"]
    assert len(rows) == len(expected) == 32 * 6
    assert sorted(map(tuple, rows)) == sorted(
        zip(
            expected.subject.tolist(),
            expected.start.tolist(),
            expected.label.tolist(),
            expected.repetition.tolist(),
            strict=True,
        )
    )
    # Membership is D44's function of the window count alone.
    assert artifact["assignment"]["inner_fold"] == f2_fold_membership(len(rows))
    ledger = artifact["source_pool_identity"]["recordings"]
    assert [r["subject_id"] for r in ledger] == list(manifest.training_subjects)
    assert all(r["num_samples"] == 1200 for r in ledger)
    assert artifact["source_pool_sha256"] == f2_source_pool_sha256(f2_dir)


def test_f2_signal_changes_provenance_and_locator_but_not_membership(
    f2_dir: Path, f2_copy: Path
) -> None:
    subject = SplitManifest.load().training_subjects[3]

    def shifted(r):
        return type(r)(r.subject, r.emg + np.float32(0.5), r.label, r.repetition)

    _rewrite_recording(f2_copy, subject, shifted)
    before, after = build_f2_assignment(f2_dir), build_f2_assignment(f2_copy)
    assert before["rng"] == after["rng"]
    assert before["assignment"] == after["assignment"]
    assert before["assignment_sha256"] == after["assignment_sha256"]
    assert before["source_pool_sha256"] != after["source_pool_sha256"]
    changed = [
        a["subject_id"]
        for a, b in zip(
            before["source_pool_identity"]["recordings"],
            after["source_pool_identity"]["recordings"],
            strict=True,
        )
        if a != b
    ]
    assert changed == [subject]


def test_f2_label_values_change_provenance_not_membership(f2_dir: Path, f2_copy: Path) -> None:
    subject = SplitManifest.load().training_subjects[0]

    def relabelled(r):
        # The same segment boundaries, so the same window starts, with other values.
        active = r.label != 0
        label, repetition = r.label.copy(), r.repetition.copy()
        label[active], repetition[active] = 2, 4
        return type(r)(r.subject, r.emg, label, repetition)

    _rewrite_recording(f2_copy, subject, relabelled)
    before, after = build_f2_assignment(f2_dir), build_f2_assignment(f2_copy)
    assert before["assignment"]["inner_fold"] == after["assignment"]["inner_fold"]
    assert before["assignment_sha256"] == after["assignment_sha256"]
    assert before["source_pool_sha256"] != after["source_pool_sha256"]


def test_f2_never_reads_held_out_subjects_and_refuses_a_missing_development_one(
    f2_dir: Path, f2_copy: Path
) -> None:
    held_out = SplitManifest.load().test_subjects[0]
    save_processed(recording(held_out, seed=7), processed_path(f2_copy, held_out))
    assert f2_source_pool_sha256(f2_copy) == f2_source_pool_sha256(f2_dir)
    missing = processed_path(f2_copy, SplitManifest.load().training_subjects[-1])
    missing.unlink()
    with pytest.raises(FileNotFoundError, match=missing.name):
        build_f2_assignment(f2_copy)


def test_f2_refuses_a_recording_stored_under_another_subjects_name(f2_copy: Path) -> None:
    first, second = SplitManifest.load().training_subjects[:2]
    shutil.copyfile(processed_path(f2_copy, second), processed_path(f2_copy, first))
    with pytest.raises(NinaProFormatError, match="filename identifies subject"):
        build_f2_assignment(f2_copy)


def test_f2_stored_window_table_and_recording_ledger_must_agree(
    f2_dir: Path, tmp_path: Path
) -> None:
    original = build_f2_assignment(f2_dir)
    tampered = []
    duplicated = copy.deepcopy(original)
    duplicated["assignment"]["windows"][1][:2] = duplicated["assignment"]["windows"][0][:2]
    tampered.append(duplicated)
    short = copy.deepcopy(original)
    short["source_pool_identity"]["recordings"][0]["num_samples"] = 500
    tampered.append(short)
    dropped = copy.deepcopy(original)
    del dropped["source_pool_identity"]["recordings"][-1]
    tampered.append(dropped)
    for artifact in tampered:
        _refresh(artifact, material=True, source=True)
        with pytest.raises(ValueError):
            save_assignment(tmp_path / "bad.json", artifact)
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("subject", [5, 10])
@pytest.mark.parametrize("schedule", range(5))
def test_g3_all_prefixes_cover_classes_and_balance_repetitions(subject: int, schedule: int) -> None:
    artifact = build_g3_schedule(_g3_source(subject), schedule)
    trials = artifact["assignment"]["trials"]
    assert trials == g3_trial_order(subject, schedule)
    assert artifact["assignment"]["k_grid"] == list(G3_K_GRID)
    assert sorted(g for g, _ in trials[:17]) == list(range(1, 18))
    assert sorted(trials) == [[g, r] for g in range(1, 18) for r in (1, 4)]
    assert trials[0][1] == (1 if (subject + schedule) % 2 == 0 else 4)
    for k in range(35):
        counts = [sum(r == rep for _, r in trials[:k]) for rep in (1, 4)]
        assert abs(counts[0] - counts[1]) <= 1
    assert [sum(r == rep for _, r in trials[:20]) for rep in (1, 4)] == [10, 10]


def test_g3_golden_three_permutation_call_order_locked_numpy_2_4_6() -> None:
    artifact = build_g3_schedule(_g3_source(5), 0)
    assert artifact["rng"]["schedule"]["derived_seed"] == 2693202415252833913
    assert artifact["assignment"]["trials"] == G3_GOLDEN_SUBJECT5_SCHEDULE0


def test_g3_source_signal_changes_identity_but_not_trials_or_seed() -> None:
    source = _g3_source(5)
    before = build_g3_schedule(source, 0)
    source.emg[(source.label == 1) & (source.repetition == 1)] += 0.5
    after = build_g3_schedule(source, 0)
    assert before["rng"] == after["rng"]
    assert before["assignment_sha256"] == after["assignment_sha256"]
    assert before["source_pool_sha256"] != after["source_pool_sha256"]
    source.emg[source.repetition == 2] += 1
    assert trial_pool_sha256(source) == after["source_pool_sha256"]


def test_g3_refuses_missing_split_or_unwindowable_real_trial_slots() -> None:
    source = _g3_source(5)
    active = np.flatnonzero((source.label == 1) & (source.repetition == 1))
    source.label[active[len(active) // 2]] = 0
    with pytest.raises(ValueError, match="complete windowable segment"):
        build_g3_schedule(source, 0)
    short = recording(5, gestures=17, repetitions=6, samples_per_segment=399)
    with pytest.raises(ValueError, match="complete windowable segment"):
        build_g3_schedule(short, 0)


@pytest.mark.parametrize("schedule", [-1, 5, True, 1.5])
def test_g3_refuses_invalid_schedule_indices(schedule: int) -> None:
    with pytest.raises(ValueError, match="schedule_index"):
        build_g3_schedule(_g3_source(5), schedule)
    with pytest.raises(ValueError, match="schedule_index"):
        g3_trial_order(5, schedule)


@pytest.mark.parametrize("kind", ["f2", "g3"])
def test_artifact_round_trip_exclusive_creation_and_current_source_binding(
    kind: str,
    f2_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _g3_source(5)
    artifact = build_f2_assignment(f2_dir) if kind == "f2" else build_g3_schedule(source, 0)
    expected = f2_source_pool_sha256(f2_dir) if kind == "f2" else trial_pool_sha256(source)
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


@pytest.mark.parametrize("umask", [0o022, 0o002, 0o077])
def test_saved_assignments_take_the_mode_the_umask_gives_new_files(
    umask: int, tmp_path: Path
) -> None:
    artifact = build_g3_schedule(_g3_source(5), 0)
    previous = os.umask(umask)
    try:
        save_assignment(tmp_path / "schedule.json", artifact)
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "schedule.json").stat().st_mode) == 0o666 & ~umask


def test_loader_accepts_original_numpy_version_without_changing_assignment(
    f2_dir: Path, tmp_path: Path
) -> None:
    artifact = build_f2_assignment(f2_dir)
    artifact["rng"]["outer"]["numpy_version"] = "older-library-version-test-double"
    _refresh(artifact)
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
    f2_dir: Path, tmp_path: Path
) -> None:
    original = build_f2_assignment(f2_dir)
    broken = copy.deepcopy(original)
    broken["assignment"]["inner_fold"][0] = 9
    with pytest.raises(ValueError, match="checksum"):
        save_assignment(tmp_path / "bad.json", broken)
    _refresh(broken)
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
    source = _g3_source(5)
    path = tmp_path / "schedule1.json"
    digest = save_assignment(path, build_g3_schedule(source, 1))
    with pytest.raises(ValueError, match="schedule identity"):
        load_assignment(
            path,
            expected_source_pool_sha256=trial_pool_sha256(source),
            expected_file_sha256=digest,
            expected_schedule_index=0,
        )


def test_trusted_file_digest_refuses_self_consistently_rewritten_membership(
    f2_dir: Path, tmp_path: Path
) -> None:
    artifact = build_f2_assignment(f2_dir)
    path = tmp_path / "assignment.json"
    trusted = save_assignment(path, artifact)
    folds = artifact["assignment"]["inner_fold"]
    folds[0], folds[1] = folds[1], folds[0]
    _refresh(artifact, material=True)
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="trusted artifact"):
        load_assignment(
            path,
            expected_source_pool_sha256=artifact["source_pool_sha256"],
            expected_file_sha256=trusted,
        )

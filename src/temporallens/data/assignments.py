"""D44/D45's material F2 partitions and G3/G4 trial schedules.

These are bounded assignment builders, not training/evaluation runners. They share D43's
seed factory but keep experiment policy here: F2 intentionally splits individual windows;
G3 spends complete active trials and preserves class coverage and repetition balance.
Artifacts retain stable IDs, source-pool fingerprints and RNG provenance. Loading checks
the saved assignment rather than redrawing it under a potentially different NumPy version.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from temporallens.data.ninapro import SubjectRecording, load_processed, processed_path
from temporallens.data.splits import SplitManifest
from temporallens.data.windows import WindowIndex, concatenate, index_recording, segment_spans
from temporallens.utils.randomness import (
    ASSIGNMENT_PRNG,
    CONTRACT_VERSION,
    assignment_rng,
    randomization_provenance,
)

F2_KIND = "f2_window_assignment_v1"
G3_KIND = "g3_trial_schedule_v1"
F2_SCOPE = "f2_random_window"
F2_OUTER_PURPOSE = "outer_test_assignment"
F2_INNER_PURPOSE = "inner_calibration_folds"
WINDOW_SIZE = 400  # D24 windows; G3 also requires every eligible trial to hold one
F2_STRIDE = 100
G3_SCOPE = "g3_g4_personalization"
G3_PURPOSE = "acquisition_schedule"
G3_K_GRID = (0, 1, 2, 5, 10, 17, 20, 34)
G3_SCHEDULE_COUNT = 5
# D46: the Git-tracked trust anchor, repository-relative as the configs name it.
ASSIGNMENT_REGISTRY = "configs/splits/assignment_registry.yaml"
_RECORDING_DIGEST = "sha256_v1_emg_f4_label_i2_repetition_i2"

_F2_EXECUTION = "evaluation.temperature_fit.execution"
# Config leaves that must equal what this module executes. A runner calls check_*_config before
# building or loading, so a config edited away from the versioned protocol fails instead of
# being silently ignored by code that never read it.
F2_CONFIG_POLICY: dict[str, Any] = {
    "experiment.seed": 42,
    "dataset.split": "random_window",
    "dataset.split_manifest": "configs/splits/subject_independent_v1.yaml",
    "dataset.subject_pool": "split_manifest.training_subjects",
    "dataset.test_fraction": 0.2,
    "dataset.window_size": WINDOW_SIZE,
    "dataset.stride": F2_STRIDE,
    "evaluation.temperature_fit.folds": 8,
    f"{_F2_EXECUTION}.status": "ready",
    f"{_F2_EXECUTION}.fold_assignment_algorithm": "d44_global_window_split_v1",
    f"{_F2_EXECUTION}.source_pool_construction": "manifest_development_recordings_uncapped",
    f"{_F2_EXECUTION}.source_recording_digest": _RECORDING_DIGEST,
    f"{_F2_EXECUTION}.canonical_window_order": ["subject_id", "start_sample"],
    f"{_F2_EXECUTION}.outer_test_count": "ceil_n_windows_divided_by_5",
    f"{_F2_EXECUTION}.outer_membership": "prefix_of_one_global_permutation",
    f"{_F2_EXECUTION}.inner_input_order": "recanonicalized_non_test_windows",
    f"{_F2_EXECUTION}.inner_membership": "one_independent_permutation_then_numpy_array_split_8",
    f"{_F2_EXECUTION}.stratification": "none",
    f"{_F2_EXECUTION}.composition_triggered_redraw": "forbidden",
    f"{_F2_EXECUTION}.seed_derivation": CONTRACT_VERSION,
    f"{_F2_EXECUTION}.prng": ASSIGNMENT_PRNG,
    f"{_F2_EXECUTION}.fold_index_base": 0,
    f"{_F2_EXECUTION}.component_stream_keys": {
        "scope": F2_SCOPE,
        "outer": {"purpose": F2_OUTER_PURPOSE, "indices": {}},
        "inner": {"purpose": F2_INNER_PURPOSE, "indices": {}},
    },
    f"{_F2_EXECUTION}.require_trusted_file_sha256": True,
    f"{_F2_EXECUTION}.assignment_registry": ASSIGNMENT_REGISTRY,
    f"{_F2_EXECUTION}.creation_verification": "regenerate_under_recorded_numpy_version",
}
_G3_SELECTION = "personalization.trial_selection"
G3_CONFIG_POLICY: dict[str, Any] = {
    "experiment.seed": 42,
    "dataset.split_manifest": "configs/splits/subject_independent_v1.yaml",
    "dataset.window_size": WINDOW_SIZE,
    "personalization.calibration_trials": list(G3_K_GRID),
    "personalization.k_unit": "complete_active_gesture_trial",
    "personalization.schedules_per_subject": G3_SCHEDULE_COUNT,
    f"{_G3_SELECTION}.schedule": "nested_class_aware_prefix",
    f"{_G3_SELECTION}.first_pass": "one_trial_per_gesture",
    f"{_G3_SELECTION}.second_pass": "other_eligible_repetition",
    f"{_G3_SELECTION}.seed_derivation": CONTRACT_VERSION,
    f"{_G3_SELECTION}.randomization_scope": G3_SCOPE,
    f"{_G3_SELECTION}.randomization_purpose": G3_PURPOSE,
    f"{_G3_SELECTION}.seed_indices": ["subject_id", "schedule_index"],
    f"{_G3_SELECTION}.first_pass_repetition_assignment": (
        "alternating_8_9_flip_by_subject_plus_schedule_parity"
    ),
    f"{_G3_SELECTION}.shared_across_strategies": True,
    f"{_G3_SELECTION}.reproducibility.status": "ready",
    f"{_G3_SELECTION}.reproducibility.prng": ASSIGNMENT_PRNG,
    f"{_G3_SELECTION}.reproducibility.schedule_index_base": 0,
    f"{_G3_SELECTION}.reproducibility.permutation_algorithm": (
        "d45_three_sequential_numpy_permutations"
    ),
    f"{_G3_SELECTION}.reproducibility.permutation_call_order": [
        "ascending_active_gestures",
        "ascending_complement_rep_1_gestures",
        "ascending_complement_rep_4_gestures",
    ],
    f"{_G3_SELECTION}.reproducibility.second_pass_order": (
        "shuffled_complement_groups_interleaved_underrepresented_repetition_first"
    ),
    f"{_G3_SELECTION}.reproducibility.parity_to_starting_repetition": {"even": 1, "odd": 4},
    f"{_G3_SELECTION}.reproducibility.require_trusted_file_sha256": True,
    f"{_G3_SELECTION}.reproducibility.assignment_registry": ASSIGNMENT_REGISTRY,
    f"{_G3_SELECTION}.reproducibility.creation_verification": (
        "regenerate_under_recorded_numpy_version"
    ),
}


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _integer(value: Any, low: int, high: int, field: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low}, {high}]")


def _root(root_seed: int) -> None:
    _integer(root_seed, 42, 42, "D43 experiment root seed")


def _policy_mismatches(config: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    mismatches = []
    for path, expected in policy.items():
        node: Any = config
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                mismatches.append(f"{path}: missing (protocol requires {_json(expected)})")
                break
            node = node[part]
        else:
            # JSON comparison keeps True distinct from 1 and 0.2 exact, as YAML parsed it.
            if _json(node) != _json(expected):
                mismatches.append(f"{path}: {_json(node)} (protocol requires {_json(expected)})")
    return mismatches


def check_f2_config(config: Mapping[str, Any]) -> None:
    """Refuse an F2 config whose assignment fields disagree with what this module executes.

    The builder does not read these values; it implements one versioned protocol. Comparing
    them is what turns a config edit into an error rather than a silent no-op.
    """
    mismatches = _policy_mismatches(config, F2_CONFIG_POLICY)
    if mismatches:
        raise ValueError("F2 config disagrees with D44/D46:\n  " + "\n  ".join(mismatches))


def check_g3_config(config: Mapping[str, Any]) -> None:
    """Refuse a G3/G4 config whose schedule fields disagree with what this module executes."""
    mismatches = _policy_mismatches(config, G3_CONFIG_POLICY)
    if mismatches:
        raise ValueError("G3/G4 config disagrees with D45/D46:\n  " + "\n  ".join(mismatches))


def _window_pool(index: WindowIndex, stride: int) -> tuple[dict[str, Any], list[list[int]]]:
    _integer(stride, F2_STRIDE, F2_STRIDE, "D44 source stride")
    _integer(index.window_size, WINDOW_SIZE, WINDOW_SIZE, "D44 window_size")
    columns = (index.subject, index.start, index.label, index.repetition)
    if any(c.ndim != 1 or c.dtype.kind not in "iu" for c in columns):
        raise ValueError("window columns must be one-dimensional integer arrays")
    if not index.start.size or len({c.size for c in columns}) != 1:
        raise ValueError("window columns must be nonempty and have matching lengths")
    manifest = SplitManifest.load()
    if set(index.subject.tolist()) != set(manifest.training_subjects):
        raise ValueError("F2 requires exactly the manifest's 32 development subjects")
    if np.any(index.start < 0) or np.any(index.start > 2**63 - 401):
        raise ValueError("window starts must be nonnegative int64 sample offsets")
    if np.any((index.label < 0) | (index.label > 17)):
        raise ValueError("window labels must be in [0, 17]")
    if np.any((index.repetition < 0) | (index.repetition > 6)) or np.any(
        (index.label != 0) & (index.repetition == 0)
    ):
        raise ValueError("active repetitions must be 1–6; rest may also use repetition 0")
    order = np.lexsort((index.start, index.subject))
    rows = np.column_stack([c[order].astype(np.int64) for c in columns]).tolist()
    if any(a[:2] == b[:2] for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("duplicate (subject_id, start_sample) window identity")
    identity = {
        "schema_version": 1,
        "dataset": "ninapro_db2",
        "exercise": "B",
        "manifest_hash": manifest.manifest_hash,
        "subject_ids": list(manifest.training_subjects),
        "window_size": WINDOW_SIZE,
        "stride": stride,
        "fields": ["subject_id", "start_sample", "restimulus", "rerepetition"],
    }
    return identity, rows


def _recording_sha256(recording: SubjectRecording) -> str:
    # Every stream the index and a future fit consume, in a fixed dtype/shape-tagged encoding,
    # so the digest does not depend on how the .npz archive happened to be compressed.
    digest = hashlib.sha256()
    for name, values, dtype in (
        ("emg", recording.emg, "<f4"),
        ("label", recording.label, "<i2"),
        ("repetition", recording.repetition, "<i2"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(f"{name}:{dtype}:{list(array.shape)}\n".encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _f2_source(data_dir: str | Path) -> tuple[dict[str, Any], list[list[int]]]:
    manifest = SplitManifest.load()
    indices, recordings = [], []
    # One recording at a time: only its index and digest outlive the loop, never the signal.
    for subject in manifest.training_subjects:
        path = processed_path(data_dir, subject)
        if not path.is_file():
            raise FileNotFoundError(
                f"F2 needs all 32 development subjects; {path} is missing. "
                "Run scripts/prepare_dataset.py for the manifest's training subjects."
            )
        recording = load_processed(path)
        indices.append(index_recording(recording, window_size=WINDOW_SIZE, stride=F2_STRIDE))
        recordings.append(
            {
                "subject_id": subject,
                "num_samples": recording.num_samples,
                "sha256": _recording_sha256(recording),
            }
        )
        del recording
    identity, rows = _window_pool(concatenate(indices), F2_STRIDE)
    identity["recording_digest"] = _RECORDING_DIGEST
    identity["recordings"] = recordings
    return identity, rows


def _trial_pool(recording: SubjectRecording) -> tuple[dict[str, Any], list[list[int]]]:
    _integer(recording.subject, 1, 40, "subject_id")
    manifest = SplitManifest.load()
    rows = sorted(
        [int(recording.label[start]), int(recording.repetition[start]), start, stop]
        for start, stop in segment_spans(recording.label, recording.repetition)
        if recording.label[start] != 0
        and recording.repetition[start] in manifest.calibration_repetitions
    )
    expected = [[g, r] for g in range(1, 18) for r in manifest.calibration_repetitions]
    if [row[:2] for row in rows] != expected or any(
        stop - start < WINDOW_SIZE for _, _, start, stop in rows
    ):
        raise ValueError(
            "G3 requires one complete windowable segment for each of 34 eligible trials"
        )
    signal_hash = hashlib.sha256()
    for _, _, start, stop in rows:
        signal_hash.update(np.ascontiguousarray(recording.emg[start:stop], dtype="<f4").tobytes())
    identity = {
        "schema_version": 1,
        "dataset": "ninapro_db2",
        "exercise": "B",
        "manifest_hash": manifest.manifest_hash,
        "subject_id": recording.subject,
        "fields": ["gesture", "rerepetition", "start_sample", "stop_sample"],
        "calibration_repetitions": list(manifest.calibration_repetitions),
        "calibration_signal_sha256": signal_hash.hexdigest(),
    }
    return identity, rows


def f2_source_pool_sha256(data_dir: str | Path) -> str:
    """Fingerprint F2's current inputs without drawing an assignment.

    Reads the manifest's 32 development recordings from ``data_dir`` exactly as
    build_f2_assignment does, so the result is the expected source digest for loading or
    registering an assignment. Held-out subjects' files are never opened.
    """
    identity, rows = _f2_source(data_dir)
    return _digest({"identity": identity, "ids": rows})


def trial_pool_sha256(recording: SubjectRecording) -> str:
    """Fingerprint the actual eligible trial segments and signal without random draws.

    The manifest binds repetitions 1/4. Missing, duplicate, split or shorter-than-window
    active segments are refused; both boundaries and canonical float32 signal values
    enter the source identity. Evaluation-repetition signal values are not inspected.
    """
    identity, rows = _trial_pool(recording)
    return _digest({"identity": identity, "ids": rows})


def _artifact(
    kind: str,
    identity: dict[str, Any],
    pool: list[list[int]],
    assignment: dict[str, Any],
    rng: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "schema_version": 1,
        "kind": kind,
        "manifest_hash": identity["manifest_hash"],
        "source_pool_identity": identity,
        "source_pool_sha256": _digest({"identity": identity, "ids": pool}),
        "rng": rng,
        "assignment": assignment,
    }
    out["assignment_sha256"] = _digest(_material(kind, assignment))
    # The material assignment, its provenance envelope and the exact serialized file
    # have distinct identities. The last is returned by save_assignment for run logging.
    out["artifact_sha256"] = _digest(out)
    return out


def _material(kind: str, assignment: dict[str, Any]) -> dict[str, Any]:
    if kind == F2_KIND:
        return {
            "window_ids": [r[:2] for r in assignment["windows"]],
            "inner_fold": assignment["inner_fold"],
        }
    return {"subject_id": assignment["subject_id"], "trials": assignment["trials"]}


def f2_fold_membership(n_windows: int, *, root_seed: int = 42) -> list[int]:
    """D44's membership of ``n_windows`` canonically ordered windows: -1 outer test, 0–7 folds.

    It depends only on the count and the two fixed D43 keys, never on labels, repetitions or
    signal. The first ceil(N/5) positions of one global permutation become test windows; the
    re-sorted remainder is independently permuted and split into eight chunks, earlier fold
    IDs receiving any remainder. Creation-time verification calls this to re-derive a saved
    assignment.
    """
    _root(root_seed)
    if type(n_windows) is not int or n_windows < 0:
        raise ValueError("n_windows must be a nonnegative integer")
    n_test = (n_windows + 4) // 5
    outer = assignment_rng(root_seed, F2_SCOPE, F2_OUTER_PURPOSE, {}).permutation(n_windows)
    non_test = np.sort(outer[n_test:])
    if non_test.size < 8:
        raise ValueError("F2 requires at least eight non-test windows")
    inner = assignment_rng(root_seed, F2_SCOPE, F2_INNER_PURPOSE, {}).permutation(non_test)
    folds = np.full(n_windows, -1, dtype=np.int8)
    for fold, members in enumerate(np.array_split(inner, 8)):
        folds[members] = fold
    return folds.tolist()


def build_f2_assignment(data_dir: str | Path, *, root_seed: int = 42) -> dict[str, Any]:
    """Build D44's fixed outer 20% and eight inner random-window folds from F2's real inputs.

    The 32 development subjects come from the manifest, and the uncapped 400/100 index is
    built here, one recording at a time, so a caller cannot supply a subsampled or
    differently windowed index. Each recording's content digest enters the source identity:
    a changed signal changes provenance and the artifact's locator, never membership. Rows
    remain canonically ordered; assignment order is not a training-shuffle contract.
    """
    _root(root_seed)
    identity, rows = _f2_source(data_dir)
    folds = f2_fold_membership(len(rows), root_seed=root_seed)
    return _artifact(
        F2_KIND,
        identity,
        rows,
        {
            "windows": rows,
            "inner_fold": folds,
            "outer_test_count": (len(rows) + 4) // 5,
            "inner_fold_counts": [folds.count(i) for i in range(8)],
        },
        {
            "outer": randomization_provenance(root_seed, F2_SCOPE, F2_OUTER_PURPOSE, {}),
            "inner": randomization_provenance(root_seed, F2_SCOPE, F2_INNER_PURPOSE, {}),
        },
    )


def g3_trial_order(subject_id: int, schedule_index: int, *, root_seed: int = 42) -> list[list[int]]:
    """D45's ordered 34 ``[gesture, rerepetition]`` trials for one subject and schedule.

    One local generator performs three permutations in order: ascending gesture IDs,
    ascending gestures needing repetition 1, ascending gestures needing repetition 4.
    First-pass repetitions alternate, with even subject+schedule parity starting at 1.
    The shuffled complementary groups interleave, starting with the underrepresented
    repetition. Every k-prefix has repetition counts differing by at most one. The order
    depends on nothing in the recording; build_g3_schedule binds it to one.
    """
    _root(root_seed)
    _integer(subject_id, 1, 40, "subject_id")
    _integer(schedule_index, 0, G3_SCHEDULE_COUNT - 1, "schedule_index")
    indices = {"subject_id": subject_id, "schedule_index": schedule_index}
    rng = assignment_rng(root_seed, G3_SCOPE, G3_PURPOSE, indices)
    order = rng.permutation(np.arange(1, 18, dtype=np.int64)).tolist()
    starting = 1 if (subject_id + schedule_index) % 2 == 0 else 4
    first = [[g, starting if i % 2 == 0 else 5 - starting] for i, g in enumerate(order)]
    groups = {
        r: rng.permutation(sorted(g for g, used in first if used != r)).tolist() for r in (1, 4)
    }
    underrepresented = 5 - starting
    second = []
    for i in range(17):
        repetition = underrepresented if i % 2 == 0 else starting
        second.append([groups[repetition][i // 2], repetition])
    return first + second


def build_g3_schedule(
    recording: SubjectRecording, schedule_index: int, *, root_seed: int = 42
) -> dict[str, Any]:
    """Build D45's 34-trial schedule, shared unchanged by G3/G4 and all strategies.

    The order comes from g3_trial_order; this binds it to the recording's actual eligible
    segments and calibration signal. This is offline trial selection, not DB2 recording
    chronology or cumulative model training.
    """
    _root(root_seed)
    _integer(schedule_index, 0, G3_SCHEDULE_COUNT - 1, "schedule_index")
    identity, pool = _trial_pool(recording)
    subject_id = recording.subject
    indices = {"subject_id": subject_id, "schedule_index": schedule_index}
    return _artifact(
        G3_KIND,
        identity,
        pool,
        {
            "subject_id": subject_id,
            "schedule_index": schedule_index,
            "trials": g3_trial_order(subject_id, schedule_index, root_seed=root_seed),
            "source_trials": pool,
            "k_grid": list(G3_K_GRID),
        },
        {"schedule": randomization_provenance(root_seed, G3_SCOPE, G3_PURPOSE, indices)},
    )


def _check_rng(record: Any, expected: dict[str, Any]) -> None:
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError("invalid RNG provenance fields")
    if not isinstance(record["numpy_version"], str) or not record["numpy_version"]:
        raise ValueError("missing original NumPy version")
    # Artifact assignments remain authoritative across library versions. Validate the key
    # and seed without demanding a redraw under the currently installed NumPy version.
    for field in ("key", "derived_seed", "assignment_prng"):
        if _json(record[field]) != _json(expected[field]):
            raise ValueError(f"RNG provenance mismatch: {field}")


def _f2_recordings(identity: Any, manifest: SplitManifest) -> list[dict[str, Any]]:
    recordings = identity.get("recordings") if isinstance(identity, dict) else None
    if (
        not isinstance(recordings, list)
        or [r.get("subject_id") if isinstance(r, dict) else None for r in recordings]
        != list(manifest.training_subjects)
        or any(
            set(r) != {"subject_id", "num_samples", "sha256"}
            or type(r["num_samples"]) is not int
            or r["num_samples"] < WINDOW_SIZE
            or not _is_sha256(r["sha256"])
            for r in recordings
        )
    ):
        raise ValueError("invalid F2 source-recording ledger")
    return recordings


def _checked(artifact: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "kind",
        "manifest_hash",
        "source_pool_identity",
        "source_pool_sha256",
        "rng",
        "assignment",
        "assignment_sha256",
        "artifact_sha256",
    }
    if not isinstance(artifact, dict) or set(artifact) != fields:
        raise ValueError("invalid assignment artifact fields")
    _integer(artifact["schema_version"], 1, 1, "assignment schema_version")
    payload = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if artifact["artifact_sha256"] != _digest(payload):
        raise ValueError("artifact checksum mismatch")
    manifest = SplitManifest.load()
    if artifact["manifest_hash"] != manifest.manifest_hash:
        raise ValueError("assignment manifest mismatch")
    assignment, rng = artifact["assignment"], artifact["rng"]
    if not isinstance(assignment, dict) or not isinstance(rng, dict):
        raise ValueError("assignment and RNG provenance must be mappings")
    if artifact["kind"] == F2_KIND:
        if set(assignment) != {"windows", "inner_fold", "outer_test_count", "inner_fold_counts"}:
            raise ValueError("invalid F2 assignment fields")
        rows = assignment["windows"]
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(row, list) or len(row) != 4 or any(type(v) is not int for v in row)
                for row in rows
            )
        ):
            raise ValueError("invalid F2 window table")
        if any(
            not 1 <= row[0] <= 40
            or not 0 <= row[1] <= 2**63 - 401
            or not 0 <= row[2] <= 17
            or not 0 <= row[3] <= 6
            for row in rows
        ):
            raise ValueError("invalid window metadata range")
        array = np.asarray(rows, dtype=np.int64)
        identity, pool = _window_pool(
            WindowIndex(
                array[:, 0].astype(np.int16),
                array[:, 1],
                array[:, 2],
                array[:, 3].astype(np.int16),
                window_size=WINDOW_SIZE,
            ),
            stride=F2_STRIDE,
        )
        if pool != rows:
            raise ValueError("F2 artifact windows are not in canonical order")
        recordings = _f2_recordings(artifact["source_pool_identity"], manifest)
        lengths = {r["subject_id"]: r["num_samples"] for r in recordings}
        if any(row[1] + WINDOW_SIZE > lengths[row[0]] for row in rows):
            raise ValueError("F2 window extends past its recorded subject length")
        identity["recording_digest"] = _RECORDING_DIGEST
        identity["recordings"] = recordings
        folds = assignment["inner_fold"]
        if (
            not isinstance(folds, list)
            or len(folds) != len(rows)
            or any(type(v) is not int or not -1 <= v < 8 for v in folds)
        ):
            raise ValueError("invalid F2 fold assignment")
        n_test = (len(rows) + 4) // 5
        q, r = divmod(len(rows) - n_test, 8)
        counts = [folds.count(i) for i in range(8)]
        if folds.count(-1) != n_test or counts != [q + (i < r) for i in range(8)]:
            raise ValueError("F2 partition counts do not follow D44")
        if _json(assignment["outer_test_count"]) != _json(n_test) or _json(
            assignment["inner_fold_counts"]
        ) != _json(counts):
            raise ValueError("F2 stored counts mismatch")
        expected_rng = {
            "outer": randomization_provenance(42, F2_SCOPE, F2_OUTER_PURPOSE, {}),
            "inner": randomization_provenance(42, F2_SCOPE, F2_INNER_PURPOSE, {}),
        }
    elif artifact["kind"] == G3_KIND:
        if set(assignment) != {"subject_id", "schedule_index", "trials", "source_trials", "k_grid"}:
            raise ValueError("invalid G3 schedule fields")
        subject, schedule = assignment["subject_id"], assignment["schedule_index"]
        _integer(subject, 1, 40, "subject_id")
        _integer(schedule, 0, 4, "schedule_index")
        identity, pool = artifact["source_pool_identity"], assignment["source_trials"]
        expected_identity = {
            "schema_version": 1,
            "dataset": "ninapro_db2",
            "exercise": "B",
            "manifest_hash": manifest.manifest_hash,
            "subject_id": subject,
            "fields": ["gesture", "rerepetition", "start_sample", "stop_sample"],
            "calibration_repetitions": list(manifest.calibration_repetitions),
        }
        if (
            not isinstance(identity, dict)
            or set(identity) != set(expected_identity) | {"calibration_signal_sha256"}
            or any(_json(identity[k]) != _json(v) for k, v in expected_identity.items())
        ):
            raise ValueError("invalid G3 source identity")
        if not _is_sha256(identity["calibration_signal_sha256"]):
            raise ValueError("invalid calibration signal digest")
        if (
            not isinstance(pool, list)
            or len(pool) != 34
            or any(
                not isinstance(row, list)
                or len(row) != 4
                or any(type(v) is not int for v in row)
                or row[2] < 0
                or row[3] - row[2] < WINDOW_SIZE
                for row in pool
            )
            or [row[:2] for row in pool] != [[g, r] for g in range(1, 18) for r in (1, 4)]
        ):
            raise ValueError("invalid G3 source-trial ledger")
        spans = sorted((row[2], row[3]) for row in pool)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:], strict=False)):
            raise ValueError("overlapping G3 source trials")
        trials = assignment["trials"]
        if (
            not isinstance(trials, list)
            or len(trials) != 34
            or any(
                not isinstance(t, list) or len(t) != 2 or any(type(v) is not int for v in t)
                for t in trials
            )
            or sorted(trials) != [row[:2] for row in pool]
        ):
            raise ValueError("G3 schedule must contain each of the 34 eligible trials exactly once")
        if sorted(t[0] for t in trials[:17]) != list(range(1, 18)):
            raise ValueError("G3 first pass must cover every gesture once")
        starting = 1 if (subject + schedule) % 2 == 0 else 4
        if any(t[1] != (starting if i % 2 == 0 else 5 - starting) for i, t in enumerate(trials)):
            raise ValueError("G3 repetition alternation/parity mismatch")
        if _json(assignment["k_grid"]) != _json(list(G3_K_GRID)):
            raise ValueError("G3 k grid mismatch")
        expected_rng = {
            "schedule": randomization_provenance(
                42, G3_SCOPE, G3_PURPOSE, {"subject_id": subject, "schedule_index": schedule}
            )
        }
    else:
        raise ValueError("unsupported assignment artifact kind")
    if artifact["assignment_sha256"] != _digest(_material(artifact["kind"], assignment)):
        raise ValueError("material assignment checksum mismatch")
    if _json(artifact["source_pool_identity"]) != _json(identity) or artifact[
        "source_pool_sha256"
    ] != _digest({"identity": identity, "ids": pool}):
        raise ValueError("source-pool identity/checksum mismatch")
    if set(rng) != set(expected_rng):
        raise ValueError("RNG component identity mismatch")
    for name, expected in expected_rng.items():
        _check_rng(rng[name], expected)
    return artifact


def save_assignment(path: str | Path, artifact: dict[str, Any]) -> str:
    """Atomically create a write-once-through-this-helper JSON; return its file SHA-256.

    Existing paths are never overwritten. The file takes the mode the user's umask gives any
    new file. The returned digest covers exact file bytes; register_assignment anchors it in
    the Git-tracked registry, and loading requires that recorded digest.
    """
    _checked(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_json(artifact) + "\n").encode("utf-8")
    # open(..., "x") creates exclusively with 0o666 minus the umask. tempfile would make an
    # owner-only file, and the hard link below shares that mode with the final name.
    temporary = destination.with_name(f".assignment-{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, destination)  # Atomic exclusive creation, never replace an old one.
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate assignment JSON key: {key}")
        out[key] = value
    return out


def load_assignment(
    path: str | Path,
    *,
    expected_source_pool_sha256: str,
    expected_file_sha256: str,
    expected_schedule_index: int | None = None,
) -> dict[str, Any]:
    """Verify an assignment against the current source pool, without redrawing membership.

    Compute the expected source digest via f2_source_pool_sha256 or trial_pool_sha256 from
    the current inputs, and take the exact-file digest from the assignment registry, never
    from the file being loaded. G3 additionally requires the requested schedule index; all
    five schedules share one source pool. Duplicate keys/non-finite constants are refused.
    """

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite assignment JSON constant: {value}")

    encoded = Path(path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_file_sha256:
        raise ValueError("assignment file checksum differs from the trusted artifact record")
    artifact = _checked(
        json.loads(encoded, object_pairs_hook=_unique_keys, parse_constant=invalid_constant)
    )
    if artifact["source_pool_sha256"] != expected_source_pool_sha256:
        raise ValueError("assignment does not belong to the current source pool")
    if artifact["kind"] == G3_KIND:
        _integer(expected_schedule_index, 0, 4, "expected_schedule_index")
        if artifact["assignment"]["schedule_index"] != expected_schedule_index:
            raise ValueError("assignment schedule identity mismatch")
    elif expected_schedule_index is not None:
        raise ValueError("F2 does not have an acquisition schedule index")
    return artifact

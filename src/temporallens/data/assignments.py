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
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from temporallens.data.ninapro import SubjectRecording
from temporallens.data.splits import SplitManifest
from temporallens.data.windows import WindowIndex, segment_spans
from temporallens.utils.randomness import assignment_rng, randomization_provenance

F2_SCOPE = "f2_random_window"
F2_OUTER_PURPOSE = "outer_test_assignment"
F2_INNER_PURPOSE = "inner_calibration_folds"
G3_SCOPE = "g3_g4_personalization"
G3_PURPOSE = "acquisition_schedule"
G3_K_GRID = (0, 1, 2, 5, 10, 17, 20, 34)
G3_SCHEDULE_COUNT = 5


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _integer(value: Any, low: int, high: int, field: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low}, {high}]")


def _root(root_seed: int) -> None:
    _integer(root_seed, 42, 42, "D43 experiment root seed")


def _window_pool(index: WindowIndex, stride: int) -> tuple[dict[str, Any], list[list[int]]]:
    _integer(stride, 100, 100, "D44 source stride")
    _integer(index.window_size, 400, 400, "D44 window_size")
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
        "window_size": 400,
        "stride": stride,
        "fields": ["subject_id", "start_sample", "restimulus", "rerepetition"],
    }
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
        stop - start < 400 for _, _, start, stop in rows
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


def window_pool_sha256(index: WindowIndex, *, stride: int) -> str:
    """Fingerprint canonical F2 window metadata without drawing an assignment.

    Stride is explicit because WindowIndex cannot recover it from segment-aligned starts.
    This identifies the logical window pool, not EMG values; future run provenance must
    separately bind processed recording artifacts. Input row order does not affect it.
    """
    identity, rows = _window_pool(index, stride)
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
    if kind == "f2_window_assignment_v1":
        return {
            "window_ids": [r[:2] for r in assignment["windows"]],
            "inner_fold": assignment["inner_fold"],
        }
    return {"subject_id": assignment["subject_id"], "trials": assignment["trials"]}


def build_f2_assignment(index: WindowIndex, *, stride: int, root_seed: int = 42) -> dict[str, Any]:
    """Build D44's fixed outer 20% and eight inner random-window folds.

    Canonical row IDs, not labels/repetitions, determine membership. The first ceil(N/5)
    positions of one global permutation become test windows. Re-sort the remainder,
    independently permute it, and split into eight chunks; earlier fold IDs receive any
    remainder. Fold -1 denotes outer test, 0–7 the inner folds. No stratification,
    grouping or composition-triggered redraw occurs. Rows remain canonically ordered;
    assignment order is not a training-shuffle contract.
    """
    _root(root_seed)
    identity, rows = _window_pool(index, stride)
    n_test = (len(rows) + 4) // 5
    outer = assignment_rng(root_seed, F2_SCOPE, F2_OUTER_PURPOSE, {}).permutation(len(rows))
    non_test = np.sort(outer[n_test:])
    if non_test.size < 8:
        raise ValueError("F2 requires at least eight non-test windows")
    inner = assignment_rng(root_seed, F2_SCOPE, F2_INNER_PURPOSE, {}).permutation(non_test)
    folds = np.full(len(rows), -1, dtype=np.int8)
    for fold, members in enumerate(np.array_split(inner, 8)):
        folds[members] = fold
    return _artifact(
        "f2_window_assignment_v1",
        identity,
        rows,
        {
            "windows": rows,
            "inner_fold": folds.tolist(),
            "outer_test_count": n_test,
            "inner_fold_counts": np.bincount(folds[folds >= 0], minlength=8).tolist(),
        },
        {
            "outer": randomization_provenance(root_seed, F2_SCOPE, F2_OUTER_PURPOSE, {}),
            "inner": randomization_provenance(root_seed, F2_SCOPE, F2_INNER_PURPOSE, {}),
        },
    )


def build_g3_schedule(
    recording: SubjectRecording, schedule_index: int, *, root_seed: int = 42
) -> dict[str, Any]:
    """Build D45's 34-trial schedule, shared unchanged by G3/G4 and all strategies.

    One local generator performs three permutations in order: ascending gesture IDs,
    ascending gestures needing repetition 1, ascending gestures needing repetition 4.
    First-pass repetitions alternate, with even subject+schedule parity starting at 1.
    The shuffled complementary groups interleave, starting with the underrepresented
    repetition. Every k-prefix has repetition counts differing by at most one. This is
    offline trial selection, not DB2 recording chronology or cumulative model training.
    """
    _root(root_seed)
    _integer(schedule_index, 0, G3_SCHEDULE_COUNT - 1, "schedule_index")
    identity, pool = _trial_pool(recording)
    subject_id = recording.subject
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
    return _artifact(
        "g3_trial_schedule_v1",
        identity,
        pool,
        {
            "subject_id": subject_id,
            "schedule_index": schedule_index,
            "trials": first + second,
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
    if artifact["manifest_hash"] != SplitManifest.load().manifest_hash:
        raise ValueError("assignment manifest mismatch")
    assignment, rng = artifact["assignment"], artifact["rng"]
    if not isinstance(assignment, dict) or not isinstance(rng, dict):
        raise ValueError("assignment and RNG provenance must be mappings")
    if artifact["kind"] == "f2_window_assignment_v1":
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
                window_size=400,
            ),
            stride=100,
        )
        if pool != rows:
            raise ValueError("F2 artifact windows are not in canonical order")
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
    elif artifact["kind"] == "g3_trial_schedule_v1":
        if set(assignment) != {"subject_id", "schedule_index", "trials", "source_trials", "k_grid"}:
            raise ValueError("invalid G3 schedule fields")
        subject, schedule = assignment["subject_id"], assignment["schedule_index"]
        _integer(subject, 1, 40, "subject_id")
        _integer(schedule, 0, 4, "schedule_index")
        identity, pool = artifact["source_pool_identity"], assignment["source_trials"]
        manifest = SplitManifest.load()
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
        signal_digest = identity["calibration_signal_sha256"]
        if (
            not isinstance(signal_digest, str)
            or len(signal_digest) != 64
            or any(c not in "0123456789abcdef" for c in signal_digest)
        ):
            raise ValueError("invalid calibration signal digest")
        if (
            not isinstance(pool, list)
            or len(pool) != 34
            or any(
                not isinstance(row, list)
                or len(row) != 4
                or any(type(v) is not int for v in row)
                or row[2] < 0
                or row[3] - row[2] < 400
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

    Existing paths are never overwritten. The material assignment and logical provenance
    envelope have separate checksums; this returned digest covers exact file bytes and
    must be anchored in the creation/run registry. Filesystem permissions are not an
    immutability guarantee; load_assignment requires that trusted digest explicitly.
    """
    _checked(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_json(artifact) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".assignment-", delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.link(
            temporary, destination
        )  # Atomic exclusive creation, never replace an old assignment.
    finally:
        if temporary is not None:
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

    Compute the expected hash via window_pool_sha256 or trial_pool_sha256 from the
    current inputs, not from the file being checked. The exact-file digest comes from a
    trusted creation/run record, not a fresh hash of the file being loaded. G3 additionally
    requires the requested schedule index; all five schedules share one source pool.
    Original NumPy version is retained; duplicate keys/non-finite constants are refused.
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
    if artifact["kind"] == "g3_trial_schedule_v1":
        _integer(expected_schedule_index, 0, 4, "expected_schedule_index")
        if artifact["assignment"]["schedule_index"] != expected_schedule_index:
            raise ValueError("assignment schedule identity mismatch")
    elif expected_schedule_index is not None:
        raise ValueError("F2 does not have an acquisition schedule index")
    return artifact

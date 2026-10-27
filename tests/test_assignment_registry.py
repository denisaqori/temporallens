"""D46's Git-tracked assignment registry, exercised on test doubles in ``tmp_path`` only."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
from _synthetic import recording, write_development_recordings

from temporallens.data.assignment_registry import (
    DEFAULT_REGISTRY_PATH,
    UnregisteredAssignmentError,
    load_registered_assignment,
    read_registry,
    register_assignment,
    registered_entry,
    render_registry,
)
from temporallens.data.assignments import (
    ASSIGNMENT_REGISTRY,
    F2_KIND,
    G3_KIND,
    build_f2_assignment,
    build_g3_schedule,
    f2_source_pool_sha256,
    save_assignment,
    trial_pool_sha256,
)
from temporallens.data.splits import REPO_ROOT


@pytest.fixture(scope="module")
def f2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_development_recordings(tmp_path_factory.mktemp("f2_processed"))


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(render_registry([]), encoding="utf-8")
    return path


def _g3_source(subject: int = 5):
    return recording(subject, gestures=17, repetitions=6, samples_per_segment=400)


def _resave(artifact: dict, path: Path) -> str:
    """Re-digest an edited artifact self-consistently and save it, as a buggy builder might."""

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()

    a = artifact["assignment"]
    artifact["assignment_sha256"] = digest(
        {"window_ids": [r[:2] for r in a["windows"]], "inner_fold": a["inner_fold"]}
        if artifact["kind"] == F2_KIND
        else {"subject_id": a["subject_id"], "trials": a["trials"]}
    )
    artifact["artifact_sha256"] = digest(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    )
    return save_assignment(path, artifact)


def _register(path: Path, digest: str, source: str, registry: Path, root: Path, **identity: int):
    return register_assignment(
        path,
        identity=identity,
        file_sha256=digest,
        expected_source_pool_sha256=source,
        registry_path=registry,
        root=root,
    )


def test_the_committed_registry_is_valid_canonical_and_where_the_configs_point() -> None:
    assert DEFAULT_REGISTRY_PATH == REPO_ROOT / ASSIGNMENT_REGISTRY
    entries = read_registry()
    assert DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8") == render_registry(entries)


@pytest.mark.parametrize("kind", ["f2", "g3"])
def test_register_then_load_through_the_registry(
    kind: str, f2_dir: Path, registry: Path, tmp_path: Path
) -> None:
    source = _g3_source(5)
    if kind == "f2":
        artifact, current, identity = build_f2_assignment(f2_dir), f2_source_pool_sha256(f2_dir), {}
    else:
        artifact, current = build_g3_schedule(source, 2), trial_pool_sha256(source)
        identity = {"subject_id": 5, "schedule_index": 2}
    path = tmp_path / "data" / "assignment.json"
    digest = save_assignment(path, artifact)
    entry = _register(path, digest, current, registry, tmp_path, **identity)
    assert entry == {
        "kind": artifact["kind"],
        "identity": identity,
        "source_pool_sha256": current,
        "assignment_sha256": artifact["assignment_sha256"],
        "path": "data/assignment.json",
        "file_sha256": digest,
    }
    assert read_registry(registry) == [entry]
    assert (
        registered_entry(
            artifact["kind"], identity, source_pool_sha256=current, registry_path=registry
        )
        == entry
    )
    loaded = load_registered_assignment(
        artifact["kind"],
        identity,
        source_pool_sha256=current,
        registry_path=registry,
        root=tmp_path,
    )
    assert loaded == artifact


def test_registering_the_same_artifact_again_changes_nothing(
    registry: Path, tmp_path: Path
) -> None:
    source = _g3_source(5)
    path = tmp_path / "schedule0.json"
    digest = save_assignment(path, build_g3_schedule(source, 0))
    args = (path, digest, trial_pool_sha256(source), registry, tmp_path)
    first = _register(*args, subject_id=5, schedule_index=0)
    text = registry.read_text()
    assert _register(*args, subject_id=5, schedule_index=0) == first
    assert registry.read_text() == text


def test_a_frozen_identity_is_never_replaced(registry: Path, tmp_path: Path) -> None:
    source = _g3_source(5)
    artifact = build_g3_schedule(source, 0)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    digest = save_assignment(first, artifact)
    _register(
        first, digest, trial_pool_sha256(source), registry, tmp_path, subject_id=5, schedule_index=0
    )
    text = registry.read_text()
    digest = save_assignment(second, artifact)
    with pytest.raises(ValueError, match="never replaced"):
        _register(
            second,
            digest,
            trial_pool_sha256(source),
            registry,
            tmp_path,
            subject_id=5,
            schedule_index=0,
        )
    assert registry.read_text() == text


def test_registration_rederives_f2_membership(f2_dir: Path, registry: Path, tmp_path: Path) -> None:
    artifact = build_f2_assignment(f2_dir)
    folds = artifact["assignment"]["inner_fold"]
    # Swap a fold-0 and a fold-1 window: every count and structural check still passes.
    i, j = folds.index(0), folds.index(1)
    folds[i], folds[j] = folds[j], folds[i]
    path = tmp_path / "f2.json"
    digest = _resave(artifact, path)
    with pytest.raises(ValueError, match="regenerate"):
        _register(path, digest, f2_source_pool_sha256(f2_dir), registry, tmp_path)
    assert read_registry(registry) == []


def test_registration_rederives_g3_order(registry: Path, tmp_path: Path) -> None:
    source = _g3_source(5)
    artifact = build_g3_schedule(source, 0)
    trials = artifact["assignment"]["trials"]
    # Positions 0 and 2 share a repetition, so alternation, coverage and balance all survive.
    trials[0], trials[2] = trials[2], trials[0]
    path = tmp_path / "g3.json"
    digest = _resave(artifact, path)
    with pytest.raises(ValueError, match="regenerate"):
        _register(
            path,
            digest,
            trial_pool_sha256(source),
            registry,
            tmp_path,
            subject_id=5,
            schedule_index=0,
        )


def test_registration_needs_the_numpy_version_that_drew_the_artifact(
    registry: Path, tmp_path: Path
) -> None:
    source = _g3_source(5)
    artifact = build_g3_schedule(source, 0)
    artifact["rng"]["schedule"]["numpy_version"] = "0.0.0-test-double"
    path = tmp_path / "g3.json"
    digest = _resave(artifact, path)
    with pytest.raises(ValueError, match=f"NumPy {np.__version__} is installed"):
        _register(
            path,
            digest,
            trial_pool_sha256(source),
            registry,
            tmp_path,
            subject_id=5,
            schedule_index=0,
        )


def test_registration_checks_file_source_identity_and_location(
    f2_dir: Path, registry: Path, tmp_path: Path
) -> None:
    source = _g3_source(5)
    root = tmp_path / "root"
    path = root / "g3.json"
    digest = save_assignment(path, build_g3_schedule(source, 0))
    current = trial_pool_sha256(source)
    with pytest.raises(ValueError, match="trusted artifact"):
        _register(path, "0" * 64, current, registry, root, subject_id=5, schedule_index=0)
    with pytest.raises(ValueError, match="current source pool"):
        _register(path, digest, "0" * 64, registry, root, subject_id=5, schedule_index=0)
    with pytest.raises(ValueError, match="different subject"):
        _register(path, digest, current, registry, root, subject_id=6, schedule_index=0)
    with pytest.raises(ValueError, match="schedule identity"):
        _register(path, digest, current, registry, root, subject_id=5, schedule_index=1)
    with pytest.raises(ValueError, match="outside the registry root"):
        _register(
            path, digest, current, registry, root / "elsewhere", subject_id=5, schedule_index=0
        )
    f2_path = root / "f2.json"
    f2_digest = save_assignment(f2_path, build_f2_assignment(f2_dir))
    with pytest.raises(ValueError, match="schedule index"):
        _register(
            f2_path, f2_digest, f2_source_pool_sha256(f2_dir), registry, root, schedule_index=0
        )
    assert read_registry(registry) == []


def test_changed_inputs_never_resolve_to_the_old_frozen_assignment(
    registry: Path, tmp_path: Path
) -> None:
    source = _g3_source(5)
    path = tmp_path / "g3.json"
    _register(
        path,
        save_assignment(path, build_g3_schedule(source, 0)),
        trial_pool_sha256(source),
        registry,
        tmp_path,
        subject_id=5,
        schedule_index=0,
    )
    source.emg[(source.label == 3) & (source.repetition == 4)] += 1
    with pytest.raises(UnregisteredAssignmentError) as error:
        load_registered_assignment(
            G3_KIND,
            {"subject_id": 5, "schedule_index": 0},
            source_pool_sha256=trial_pool_sha256(source),
            registry_path=registry,
            root=tmp_path,
        )
    assert isinstance(error.value, LookupError)
    with pytest.raises(UnregisteredAssignmentError):
        registered_entry(
            G3_KIND,
            {"subject_id": 5, "schedule_index": 1},
            source_pool_sha256=trial_pool_sha256(_g3_source(5)),
            registry_path=registry,
        )


def test_a_file_edited_after_freezing_is_refused(registry: Path, tmp_path: Path) -> None:
    source = _g3_source(5)
    path = tmp_path / "g3.json"
    current = trial_pool_sha256(source)
    _register(
        path,
        save_assignment(path, build_g3_schedule(source, 0)),
        current,
        registry,
        tmp_path,
        subject_id=5,
        schedule_index=0,
    )
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="trusted artifact"):
        load_registered_assignment(
            G3_KIND,
            {"subject_id": 5, "schedule_index": 0},
            source_pool_sha256=current,
            registry_path=registry,
            root=tmp_path,
        )


ENTRY = {
    "kind": G3_KIND,
    "identity": {"subject_id": 5, "schedule_index": 0},
    "source_pool_sha256": "a" * 64,
    "assignment_sha256": "b" * 64,
    "path": "data/g3.json",
    "file_sha256": "c" * 64,
}


@pytest.mark.parametrize(
    "text",
    [
        "schema_version: 1\nschema_version: 1\nassignments: []\n",
        "schema_version: 2\nassignments: []\n",
        "schema_version: 1\nassignments: []\nnote: extra\n",
        render_registry([{**ENTRY, "extra": 1}]),
        render_registry([{**ENTRY, "file_sha256": "C" * 64}]),
        render_registry([{**ENTRY, "path": "/abs/g3.json"}]),
        render_registry([{**ENTRY, "path": "data/../g3.json"}]),
        render_registry([{**ENTRY, "identity": {"subject_id": 5}}]),
        render_registry([{**ENTRY, "kind": F2_KIND}]),
        render_registry([ENTRY, {**ENTRY, "path": "data/other.json"}]),
        render_registry([ENTRY, {**ENTRY, "identity": {"subject_id": 6, "schedule_index": 0}}]),
    ],
)
def test_registry_reading_is_strict(text: str, tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_registry(path)


@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_registry_rewrites_take_the_mode_the_umask_gives_new_files(
    umask: int, registry: Path, tmp_path: Path
) -> None:
    source = _g3_source(5)
    path = tmp_path / "g3.json"
    digest = save_assignment(path, build_g3_schedule(source, 0))
    previous = os.umask(umask)
    try:
        _register(
            path,
            digest,
            trial_pool_sha256(source),
            registry,
            tmp_path,
            subject_id=5,
            schedule_index=0,
        )
    finally:
        os.umask(previous)
    assert stat.S_IMODE(registry.stat().st_mode) == 0o666 & ~umask
    assert not list(tmp_path.glob(".registry.yaml.*"))


def test_the_registry_root_resolves_paths_from_any_checkout(registry: Path, tmp_path: Path) -> None:
    # Entries are repository-relative, so a second checkout with the same data/ resolves them.
    source = _g3_source(5)
    first = tmp_path / "first"
    path = first / "data" / "g3.json"
    digest = save_assignment(path, build_g3_schedule(source, 0))
    _register(
        path, digest, trial_pool_sha256(source), registry, first, subject_id=5, schedule_index=0
    )
    second = tmp_path / "second"
    shutil.copytree(first, second)
    assert load_registered_assignment(
        G3_KIND,
        {"subject_id": 5, "schedule_index": 0},
        source_pool_sha256=trial_pool_sha256(source),
        registry_path=registry,
        root=second,
    ) == build_g3_schedule(source, 0)

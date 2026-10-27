"""D46's Git-tracked registry of frozen F2/G3 assignment artifacts.

An assignment file can be checked against a digest, but only a digest recorded somewhere the
file's own writer cannot quietly rewrite makes that check mean anything. The registry is that
place: a small YAML file beside the split manifest, committed to Git, holding each artifact's
logical identity, source-pool digest, material-assignment digest, path and exact-file digest.
The artifacts stay under ``data/`` (gitignored; they carry NinaPro labels).

Registration is the one-time freeze step. It re-derives the saved membership under the NumPy
version that produced it and refuses any discrepancy, then appends an entry. Archival loading
goes through the entry and never redraws. Non-debug runs already refuse an uncommitted tree, so
an entry a reportable run can use is necessarily one Git recorded.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from temporallens.data.assignments import (
    ASSIGNMENT_REGISTRY,
    F2_KIND,
    G3_KIND,
    G3_SCHEDULE_COUNT,
    f2_fold_membership,
    g3_trial_order,
    load_assignment,
)
from temporallens.data.splits import REPO_ROOT
from temporallens.utils.yaml_io import load_yaml

DEFAULT_REGISTRY_PATH = REPO_ROOT / ASSIGNMENT_REGISTRY
REGISTRY_SCHEMA_VERSION = 1
_ENTRY_FIELDS = (
    "kind",
    "identity",
    "source_pool_sha256",
    "assignment_sha256",
    "path",
    "file_sha256",
)
_HEADER = """\
# FROZEN ASSIGNMENT REGISTRY (D46) -- the trust anchor for F2/G3 assignment artifacts.
#
# Written only by temporallens.data.assignment_registry.register_assignment, after it has
# re-derived the artifact's membership and verified it against the current source pool. The
# artifacts live under data/ (gitignored); this file holds their identities and digests, so
# Git, not the artifact's directory, vouches for them. Entries are appended, never edited or
# removed: an assignment is frozen once. Commit this file before a run consumes an entry.
"""


class UnregisteredAssignmentError(LookupError):
    """No frozen assignment exists for this kind, identity and current source pool."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _identity(kind: Any, identity: Any) -> dict[str, int]:
    if not isinstance(identity, Mapping):
        raise ValueError("assignment identity must be a mapping")
    if kind == F2_KIND:
        if dict(identity):
            raise ValueError("F2 has exactly one assignment per source pool; identity must be {}")
        return {}
    if kind == G3_KIND:
        subject, schedule = identity.get("subject_id"), identity.get("schedule_index")
        if (
            set(identity) != {"subject_id", "schedule_index"}
            or type(subject) is not int
            or not 1 <= subject <= 40
            or type(schedule) is not int
            or not 0 <= schedule < G3_SCHEDULE_COUNT
        ):
            raise ValueError("G3 identity must be {subject_id: 1-40, schedule_index: 0-4}")
        return {"subject_id": subject, "schedule_index": schedule}
    raise ValueError(f"unsupported assignment kind {kind!r}")


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("registry path must be a POSIX string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) != value:
        raise ValueError(f"registry path must be normalized and repository-relative: {value!r}")
    return value


def _entry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != set(_ENTRY_FIELDS):
        raise ValueError(f"registry entries need exactly the fields {list(_ENTRY_FIELDS)}")
    for field in ("source_pool_sha256", "assignment_sha256", "file_sha256"):
        if not _is_sha256(raw[field]):
            raise ValueError(f"registry {field} must be a lowercase SHA-256 hex digest")
    return {
        "kind": raw["kind"],
        "identity": _identity(raw["kind"], raw["identity"]),
        "source_pool_sha256": raw["source_pool_sha256"],
        "assignment_sha256": raw["assignment_sha256"],
        "path": _relative_path(raw["path"]),
        "file_sha256": raw["file_sha256"],
    }


def _key(entry: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, int], ...], str]:
    return (entry["kind"], tuple(sorted(entry["identity"].items())), entry["source_pool_sha256"])


def read_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Every frozen entry, validated, in registration order.

    Strict YAML (duplicate keys are errors), exact fields, and at most one entry per logical
    identity and source pool, or per artifact path.
    """
    raw = load_yaml(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "assignments"}
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != REGISTRY_SCHEMA_VERSION
        or not isinstance(raw["assignments"], list)
    ):
        raise ValueError(
            f"{path}: expected schema_version {REGISTRY_SCHEMA_VERSION} and an assignments list"
        )
    entries = [_entry(item) for item in raw["assignments"]]
    if len({_key(e) for e in entries}) != len(entries):
        raise ValueError(f"{path}: two entries freeze the same identity and source pool")
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError(f"{path}: two entries name the same artifact path")
    return entries


def render_registry(entries: list[dict[str, Any]]) -> str:
    """The registry file's exact text for ``entries``: the fixed header, then the YAML body."""
    body = {"schema_version": REGISTRY_SCHEMA_VERSION, "assignments": entries}
    return _HEADER + yaml.safe_dump(body, sort_keys=False, default_flow_style=False)


def _write_registry(path: Path, entries: list[dict[str, Any]]) -> None:
    # Replace in one step with a file the umask governs; a crash leaves the old registry.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as f:
            f.write(render_registry(entries))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_regeneration(artifact: dict[str, Any]) -> None:
    versions = {record["numpy_version"] for record in artifact["rng"].values()}
    if versions != {np.__version__}:
        raise ValueError(
            f"artifact was drawn under NumPy {sorted(versions)}, but NumPy {np.__version__} is "
            "installed. Register it from the environment that created it (make setup), so its "
            "membership can be re-derived before it is frozen."
        )
    assignment = artifact["assignment"]
    if artifact["kind"] == F2_KIND:
        regenerated: Any = f2_fold_membership(len(assignment["windows"]))
        stored: Any = assignment["inner_fold"]
    else:
        regenerated = g3_trial_order(assignment["subject_id"], assignment["schedule_index"])
        stored = assignment["trials"]
    if regenerated != stored:
        raise ValueError(
            "saved membership differs from the one D44/D45 regenerate under the same NumPy "
            "version; the artifact is refused, and nothing is replaced"
        )


def register_assignment(
    path: str | Path,
    *,
    identity: Mapping[str, int],
    file_sha256: str,
    expected_source_pool_sha256: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Freeze a saved assignment: verify it, re-derive its membership, append a registry entry.

    ``file_sha256`` is the digest save_assignment returned, so a file swapped after saving is
    refused. ``expected_source_pool_sha256`` comes from the current inputs. Registering the
    identical artifact again returns its existing entry; a different artifact for an already
    frozen identity and source pool is refused, never substituted. Commit the registry after.
    """
    root_dir = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root_dir).as_posix()
    except ValueError:
        raise ValueError(f"{resolved} is outside the registry root {root_dir}") from None
    artifact = load_assignment(
        resolved,
        expected_source_pool_sha256=expected_source_pool_sha256,
        expected_file_sha256=file_sha256,
        expected_schedule_index=identity.get("schedule_index"),
    )
    logical = _identity(artifact["kind"], identity)
    if (
        artifact["kind"] == G3_KIND
        and artifact["assignment"]["subject_id"] != logical["subject_id"]
    ):
        raise ValueError("G3 artifact belongs to a different subject than the stated identity")
    _verify_regeneration(artifact)
    entry = {
        "kind": artifact["kind"],
        "identity": logical,
        "source_pool_sha256": artifact["source_pool_sha256"],
        "assignment_sha256": artifact["assignment_sha256"],
        "path": _relative_path(relative),
        "file_sha256": file_sha256,
    }
    registry = Path(registry_path)
    entries = read_registry(registry)
    for existing in entries:
        if _key(existing) == _key(entry):
            if existing == entry:
                return existing
            raise ValueError(
                "an assignment is already frozen for this identity and source pool; "
                "it is never replaced"
            )
        if existing["path"] == entry["path"]:
            raise ValueError(f"{relative} is already registered for a different assignment")
    _write_registry(registry, [*entries, entry])
    return entry


def registered_entry(
    kind: str,
    identity: Mapping[str, int],
    *,
    source_pool_sha256: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """The frozen entry for this kind, identity and current source pool.

    Raises UnregisteredAssignmentError when none exists: changed inputs have a different source
    digest, so they never resolve to an assignment frozen for the old ones.
    """
    wanted = (kind, tuple(sorted(_identity(kind, identity).items())), source_pool_sha256)
    for entry in read_registry(registry_path):
        if _key(entry) == wanted:
            return entry
    raise UnregisteredAssignmentError(
        f"no frozen {kind} assignment for {dict(identity)} and source pool "
        f"{source_pool_sha256[:12]}; build, save and register one, then commit the registry"
    )


def load_registered_assignment(
    kind: str,
    identity: Mapping[str, int],
    *,
    source_pool_sha256: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Load a frozen assignment through its registry entry, without redrawing it.

    The exact-file digest comes from the registry, the source digest from the current inputs,
    and the loaded material digest must equal the one frozen with the entry.
    """
    entry = registered_entry(
        kind, identity, source_pool_sha256=source_pool_sha256, registry_path=registry_path
    )
    artifact = load_assignment(
        Path(root) / entry["path"],
        expected_source_pool_sha256=source_pool_sha256,
        expected_file_sha256=entry["file_sha256"],
        expected_schedule_index=entry["identity"].get("schedule_index"),
    )
    if artifact["assignment_sha256"] != entry["assignment_sha256"]:
        raise ValueError(
            "loaded assignment differs from the material digest frozen in the registry"
        )
    return artifact

"""Load, verify, and hash the frozen subject-independent split manifest.

The split is *data*, not code. ``configs/splits/subject_independent_v1.yaml`` fixes which
subjects hold which role, how the training pool divides into cross-validation folds, and which
repetitions may be spent on subject calibration. It carries a SHA-256 over its own
split-defining fields, and every consumer verifies that digest before using the split.

That verification is the point. Numbers computed on different splits are not comparable, and the
language arm rests on comparing an adapter against F1's reference row — so a hand-edit that
silently moved one subject would invalidate a comparison without leaving a trace. The digest
turns that into a loud failure at load time.

Only the fields in :data:`HASHED_FIELDS` are covered. Comments and prose in the manifest may be
reworded freely; the subject assignments cannot.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "configs" / "splits" / "subject_independent_v1.yaml"

#: The fields the digest covers, in the order they are serialised.
HASHED_FIELDS = (
    "schema_version",
    "split",
    "dataset",
    "test_subjects",
    "training_subjects",
    "folds",
    "repetitions",
)

_HASH_PREFIX = "sha256:"


class SplitManifestError(RuntimeError):
    """A manifest is internally inconsistent, or its content does not match its digest."""


def canonical_json(manifest: Mapping[str, Any]) -> str:
    """Serialise the split-defining fields so the digest depends on content, not formatting."""
    missing = [field for field in HASHED_FIELDS if field not in manifest]
    if missing:
        raise SplitManifestError(f"manifest is missing required field(s): {missing}")
    payload = {field: manifest[field] for field in HASHED_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_hash(manifest: Mapping[str, Any]) -> str:
    """Return the ``sha256:<hex>`` digest of a manifest's split-defining fields."""
    digest = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


@dataclass(frozen=True)
class SplitManifest:
    """A verified split. Construct via :meth:`load`, which refuses anything inconsistent."""

    schema_version: int
    split: str
    test_subjects: tuple[int, ...]
    training_subjects: tuple[int, ...]
    fold_validation_subjects: tuple[tuple[int, ...], ...]
    calibration_repetitions: tuple[int, ...]
    evaluation_repetitions: tuple[int, ...]
    all_repetitions: tuple[int, ...]
    label_column: str
    repetition_column: str
    manifest_hash: str

    @property
    def num_folds(self) -> int:
        return len(self.fold_validation_subjects)

    def validation_subjects(self, fold: int) -> tuple[int, ...]:
        """Subjects held out for validation in ``fold``."""
        return self.fold_validation_subjects[fold]

    def training_subjects_for_fold(self, fold: int) -> tuple[int, ...]:
        """Training pool minus this fold's validation subjects.

        The complement *is* the definition — a fold's training set is never listed separately,
        so there is only one place a subject's role can be stated.
        """
        held_out = set(self.fold_validation_subjects[fold])
        return tuple(s for s in self.training_subjects if s not in held_out)

    @classmethod
    def load(cls, path: str | Path | None = None) -> SplitManifest:
        """Read a manifest, verify its digest and internal consistency, and return it."""
        manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
        raw = yaml.safe_load(manifest_path.read_text())
        if not isinstance(raw, dict):
            raise SplitManifestError(f"{manifest_path} does not contain a YAML mapping")
        _verify(raw, manifest_path)

        dataset = raw["dataset"]
        repetitions = raw["repetitions"]
        return cls(
            schema_version=raw["schema_version"],
            split=raw["split"],
            test_subjects=tuple(raw["test_subjects"]),
            training_subjects=tuple(raw["training_subjects"]),
            fold_validation_subjects=tuple(
                tuple(fold["validation_subjects"]) for fold in raw["folds"]
            ),
            calibration_repetitions=tuple(repetitions["calibration"]),
            evaluation_repetitions=tuple(repetitions["evaluation"]),
            all_repetitions=tuple(repetitions["all"]),
            label_column=dataset["label_column"],
            repetition_column=dataset["repetition_column"],
            manifest_hash=raw["manifest_hash"],
        )


def _verify(raw: Mapping[str, Any], source: Path) -> None:
    """Raise :class:`SplitManifestError` on any inconsistency. Order matters: digest last."""
    dataset = raw["dataset"]
    test_subjects: Sequence[int] = raw["test_subjects"]
    training_subjects: Sequence[int] = raw["training_subjects"]
    folds: Sequence[Mapping[str, Any]] = raw["folds"]
    repetitions = raw["repetitions"]

    test_set, training_set = set(test_subjects), set(training_subjects)
    if len(test_set) != len(test_subjects) or len(training_set) != len(training_subjects):
        raise SplitManifestError(f"{source}: a subject is listed twice within a role")
    if overlap := test_set & training_set:
        raise SplitManifestError(f"{source}: subject(s) {sorted(overlap)} are both test and train")

    expected = set(range(1, dataset["num_subjects"] + 1))
    if (assigned := test_set | training_set) != expected:
        raise SplitManifestError(
            f"{source}: subjects {sorted(expected ^ assigned)} are unassigned or out of range"
        )

    if len(folds) != raw["num_folds"]:
        raise SplitManifestError(f"{source}: num_folds disagrees with the number of fold entries")

    seen: set[int] = set()
    for fold in folds:
        validation = fold["validation_subjects"]
        if len(validation) != raw["validation_subjects_per_fold"]:
            raise SplitManifestError(
                f"{source}: fold {fold['index']} has {len(validation)} validation subjects, "
                f"expected {raw['validation_subjects_per_fold']}"
            )
        if not set(validation) <= training_set:
            raise SplitManifestError(
                f"{source}: fold {fold['index']} validates on a non-training subject"
            )
        if repeated := seen & set(validation):
            raise SplitManifestError(
                f"{source}: subject(s) {sorted(repeated)} validate in more than one fold"
            )
        seen.update(validation)

    if seen != training_set:
        raise SplitManifestError(
            f"{source}: subject(s) {sorted(training_set - seen)} never serve as validation"
        )

    calibration, evaluation = set(repetitions["calibration"]), set(repetitions["evaluation"])
    if shared := calibration & evaluation:
        raise SplitManifestError(
            f"{source}: repetition(s) {sorted(shared)} are both calibration and evaluation "
            "(generative-arm leakage rule 3)"
        )
    if not (calibration | evaluation) <= set(repetitions["all"]):
        raise SplitManifestError(f"{source}: a calibration/evaluation repetition is not in `all`")

    recorded = raw["manifest_hash"]
    actual = compute_hash(raw)
    if recorded != actual:
        raise SplitManifestError(
            f"{source}: manifest_hash does not match its contents.\n"
            f"  recorded: {recorded}\n"
            f"  actual:   {actual}\n"
            "The split changed. If that was deliberate, it needs a new versioned manifest and a "
            "DECISIONS row — the split is frozen once, never revised in place."
        )

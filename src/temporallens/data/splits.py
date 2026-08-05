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
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_MANIFEST_PATH = REPO_ROOT / "configs" / "splits" / "subject_independent_v1.yaml"
DEFAULT_MANIFEST_PATH: Path | Traversable = (
    _SOURCE_MANIFEST_PATH
    if _SOURCE_MANIFEST_PATH.is_file()
    else files("temporallens.data").joinpath("manifests", "subject_independent_v1.yaml")
)

SUPPORTED_SCHEMA_VERSION = 1
SUBJECT_INDEPENDENT_V1_HASH = (
    "sha256:a9609e264eeac9da2b5ea0f10a70a95746062841061d53cc85fba608dfff0e30"
)

_FROZEN_HASHES_BY_FILENAME = {
    "subject_independent_v1.yaml": SUBJECT_INDEPENDENT_V1_HASH,
}
_EXPECTED_DATASET = {
    "name": "ninapro_db2",
    "exercise": "B",
    "num_classes": 18,
    "num_subjects": 40,
    "label_column": "restimulus",
    "repetition_column": "rerepetition",
}
_EXPECTED_NUM_SUBJECTS = 40
_EXPECTED_NUM_FOLDS = 8
_EXPECTED_VALIDATION_SUBJECTS_PER_FOLD = 4
_EXPECTED_REPETITIONS = (1, 2, 3, 4, 5, 6)

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


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise SplitManifestError(f"unhashable YAML mapping key: {key!r}") from exc
        if duplicate:
            raise SplitManifestError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


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
        try:
            manifest_text = manifest_path.read_text()
        except (OSError, UnicodeError) as exc:
            raise SplitManifestError(f"cannot read split manifest {manifest_path}: {exc}") from exc

        try:
            raw = yaml.load(manifest_text, Loader=_UniqueKeyLoader)
        except SplitManifestError as exc:
            raise SplitManifestError(f"{manifest_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise SplitManifestError(f"{manifest_path}: invalid YAML: {exc}") from exc
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


def _require_mapping(
    parent: Mapping[str, Any], key: str, source: Path | Traversable
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise SplitManifestError(f"{source}: `{key}` must be a mapping")
    return value


def _require_int_sequence(
    parent: Mapping[str, Any], key: str, source: Path | Traversable
) -> tuple[int, ...]:
    value = parent.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SplitManifestError(f"{source}: `{key}` must be a sequence of integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise SplitManifestError(f"{source}: `{key}` must contain only integers")
    return tuple(value)


def _require_plain_int(raw: Mapping[str, Any], key: str, source: Path | Traversable) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SplitManifestError(f"{source}: `{key}` must be an integer")
    return value


def _verify(raw: Mapping[str, Any], source: Path | Traversable) -> None:
    """Raise :class:`SplitManifestError` on any inconsistency. Order matters: digest last."""
    required = {
        *HASHED_FIELDS,
        "manifest_hash",
        "num_folds",
        "validation_subjects_per_fold",
    }
    if missing := sorted(required - raw.keys()):
        raise SplitManifestError(f"{source}: manifest is missing required field(s): {missing}")

    schema_version = _require_plain_int(raw, "schema_version", source)
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise SplitManifestError(
            f"{source}: unsupported schema_version {schema_version}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    if raw["split"] != "subject_independent":
        raise SplitManifestError(
            f"{source}: unsupported split {raw['split']!r}; expected 'subject_independent'"
        )

    dataset = _require_mapping(raw, "dataset", source)
    if missing := sorted(_EXPECTED_DATASET.keys() - dataset.keys()):
        raise SplitManifestError(f"{source}: dataset is missing required field(s): {missing}")
    for key, expected_value in _EXPECTED_DATASET.items():
        if dataset[key] != expected_value:
            raise SplitManifestError(
                f"{source}: dataset.{key} is {dataset[key]!r}; expected {expected_value!r}"
            )

    test_subjects = _require_int_sequence(raw, "test_subjects", source)
    training_subjects = _require_int_sequence(raw, "training_subjects", source)

    folds_value = raw["folds"]
    if not isinstance(folds_value, Sequence) or isinstance(folds_value, (str, bytes)):
        raise SplitManifestError(f"{source}: `folds` must be a sequence of mappings")
    if any(not isinstance(fold, Mapping) for fold in folds_value):
        raise SplitManifestError(f"{source}: every fold must be a mapping")
    folds: Sequence[Mapping[str, Any]] = folds_value

    repetitions = _require_mapping(raw, "repetitions", source)
    if repetitions.get("policy") != "follow_subject":
        raise SplitManifestError(
            f"{source}: repetitions.policy is {repetitions.get('policy')!r}; "
            "expected 'follow_subject'"
        )

    test_set, training_set = set(test_subjects), set(training_subjects)
    if len(test_set) != len(test_subjects) or len(training_set) != len(training_subjects):
        raise SplitManifestError(f"{source}: a subject is listed twice within a role")
    if overlap := test_set & training_set:
        raise SplitManifestError(f"{source}: subject(s) {sorted(overlap)} are both test and train")

    expected = set(range(1, _EXPECTED_NUM_SUBJECTS + 1))
    if (assigned := test_set | training_set) != expected:
        raise SplitManifestError(
            f"{source}: subjects {sorted(expected ^ assigned)} are unassigned or out of range"
        )

    num_folds = _require_plain_int(raw, "num_folds", source)
    if num_folds != _EXPECTED_NUM_FOLDS:
        raise SplitManifestError(
            f"{source}: num_folds is {num_folds}; expected {_EXPECTED_NUM_FOLDS} (D6)"
        )
    if len(folds) != num_folds:
        raise SplitManifestError(f"{source}: num_folds disagrees with the number of fold entries")

    validation_subjects_per_fold = _require_plain_int(
        raw, "validation_subjects_per_fold", source
    )
    if validation_subjects_per_fold != _EXPECTED_VALIDATION_SUBJECTS_PER_FOLD:
        raise SplitManifestError(
            f"{source}: validation_subjects_per_fold is {validation_subjects_per_fold}; "
            f"expected {_EXPECTED_VALIDATION_SUBJECTS_PER_FOLD} (D6)"
        )

    seen: set[int] = set()
    for expected_index, fold in enumerate(folds):
        fold_index = _require_plain_int(fold, "index", source)
        if fold_index != expected_index:
            raise SplitManifestError(
                f"{source}: fold index is {fold_index}; expected contiguous index {expected_index}"
            )
        validation = _require_int_sequence(fold, "validation_subjects", source)
        if len(validation) != validation_subjects_per_fold:
            raise SplitManifestError(
                f"{source}: fold {fold_index} has {len(validation)} validation subjects, "
                f"expected {validation_subjects_per_fold}"
            )
        if len(set(validation)) != len(validation):
            raise SplitManifestError(f"{source}: fold {fold_index} lists a subject twice")
        if not set(validation) <= training_set:
            raise SplitManifestError(
                f"{source}: fold {fold_index} validates on a non-training subject"
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

    all_repetitions = _require_int_sequence(repetitions, "all", source)
    calibration_repetitions = _require_int_sequence(repetitions, "calibration", source)
    evaluation_repetitions = _require_int_sequence(repetitions, "evaluation", source)
    if all_repetitions != _EXPECTED_REPETITIONS:
        raise SplitManifestError(
            f"{source}: repetitions.all is {list(all_repetitions)}; "
            f"expected {list(_EXPECTED_REPETITIONS)}"
        )
    for name, values in (
        ("all", all_repetitions),
        ("calibration", calibration_repetitions),
        ("evaluation", evaluation_repetitions),
    ):
        if len(set(values)) != len(values):
            raise SplitManifestError(f"{source}: repetitions.{name} contains a duplicate")

    calibration, evaluation = set(calibration_repetitions), set(evaluation_repetitions)
    if shared := calibration & evaluation:
        raise SplitManifestError(
            f"{source}: repetition(s) {sorted(shared)} are both calibration and evaluation "
            "(generative-arm leakage rule 3)"
        )
    all_repetition_set = set(all_repetitions)
    if calibration | evaluation != all_repetition_set:
        missing_repetitions = sorted(all_repetition_set - (calibration | evaluation))
        extra_repetitions = sorted((calibration | evaluation) - all_repetition_set)
        raise SplitManifestError(
            f"{source}: calibration/evaluation must partition repetitions.all; "
            f"missing={missing_repetitions}, extra={extra_repetitions}"
        )

    recorded = raw["manifest_hash"]
    if not isinstance(recorded, str):
        raise SplitManifestError(f"{source}: manifest_hash must be a string")
    actual = compute_hash(raw)
    if recorded != actual:
        raise SplitManifestError(
            f"{source}: manifest_hash does not match its contents.\n"
            f"  recorded: {recorded}\n"
            f"  actual:   {actual}\n"
            "The split changed. If that was deliberate, it needs a new versioned manifest and a "
            "DECISIONS row — the split is frozen once, never revised in place."
        )

    if (pinned := _FROZEN_HASHES_BY_FILENAME.get(source.name)) and recorded != pinned:
        raise SplitManifestError(
            f"{source}: {source.name} is pinned to {pinned}, but contains {recorded}. "
            "A changed split requires a new versioned filename and DECISIONS row."
        )

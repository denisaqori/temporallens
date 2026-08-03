"""The frozen split manifest must stay frozen, and must refuse to load if it does not."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from temporallens.data.splits import (
    DEFAULT_MANIFEST_PATH,
    SplitManifest,
    SplitManifestError,
    compute_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_MANIFEST_PATH.read_text())


def _write(tmp_path: Path, manifest: dict[str, Any], *, rehash: bool = True) -> Path:
    """Write a manifest, optionally refreshing the digest so structural checks are what fail."""
    if rehash:
        manifest["manifest_hash"] = compute_hash(manifest)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


# --- the committed manifest -------------------------------------------------------------


def test_committed_manifest_loads_and_matches_its_digest() -> None:
    manifest = SplitManifest.load()
    assert manifest.split == "subject_independent"
    assert manifest.manifest_hash == compute_hash(_raw())


def test_manifest_lives_where_the_spec_says_it_does() -> None:
    assert DEFAULT_MANIFEST_PATH == REPO_ROOT / "configs" / "splits" / "subject_independent_v1.yaml"
    assert DEFAULT_MANIFEST_PATH.is_file()


def test_test_subjects_are_the_eight_the_spec_fixes() -> None:
    assert SplitManifest.load().test_subjects == (5, 10, 15, 20, 25, 30, 35, 40)


def test_roles_partition_all_forty_subjects() -> None:
    manifest = SplitManifest.load()
    assert set(manifest.test_subjects) | set(manifest.training_subjects) == set(range(1, 41))
    assert not set(manifest.test_subjects) & set(manifest.training_subjects)
    assert len(manifest.training_subjects) == 32


def test_every_training_subject_validates_exactly_once() -> None:
    manifest = SplitManifest.load()
    assert manifest.num_folds == 8
    seen = [s for fold in range(manifest.num_folds) for s in manifest.validation_subjects(fold)]
    assert sorted(seen) == sorted(manifest.training_subjects)
    assert len(seen) == len(set(seen))


def test_each_fold_trains_on_the_complement() -> None:
    manifest = SplitManifest.load()
    for fold in range(manifest.num_folds):
        training = manifest.training_subjects_for_fold(fold)
        validation = manifest.validation_subjects(fold)
        assert len(validation) == 4
        assert len(training) == 28
        assert not set(training) & set(validation)
        assert set(training) | set(validation) == set(manifest.training_subjects)
        assert not set(training) & set(manifest.test_subjects)


def test_calibration_and_evaluation_repetitions_are_disjoint() -> None:
    """Leakage rule 3: the k permitted samples never overlap the windows scored on."""
    manifest = SplitManifest.load()
    assert manifest.calibration_repetitions == (1, 4)
    assert manifest.evaluation_repetitions == (2, 3, 5, 6)
    assert not set(manifest.calibration_repetitions) & set(manifest.evaluation_repetitions)
    covered = set(manifest.calibration_repetitions) | set(manifest.evaluation_repetitions)
    assert covered == set(manifest.all_repetitions)


def test_corrected_label_columns_are_recorded() -> None:
    """D12: the uncorrected columns mislabel every movement onset."""
    manifest = SplitManifest.load()
    assert manifest.label_column == "restimulus"
    assert manifest.repetition_column == "rerepetition"


# --- the manifest must refuse to load when tampered with --------------------------------


def test_edited_split_without_rehashing_is_rejected(tmp_path: Path) -> None:
    """The case the digest exists for.

    Swapping two subjects between folds leaves a structurally perfect manifest: every subject
    still validates exactly once, every fold still holds four. No consistency check can see it.
    Only the digest can, which is why the digest is checked last — structural errors get a
    specific message, and this is what is left over.
    """
    manifest = copy.deepcopy(_raw())
    manifest["folds"][0]["validation_subjects"] = [2, 11, 21, 31]
    manifest["folds"][1]["validation_subjects"] = [1, 12, 22, 32]
    path = _write(tmp_path, manifest, rehash=False)

    with pytest.raises(SplitManifestError, match="does not match its contents"):
        SplitManifest.load(path)


def test_subject_in_two_roles_is_rejected(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_raw())
    manifest["training_subjects"].append(5)  # already a test subject
    path = _write(tmp_path, manifest)

    with pytest.raises(SplitManifestError, match="both test and train"):
        SplitManifest.load(path)


def test_unassigned_subject_is_rejected(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_raw())
    manifest["training_subjects"].remove(39)
    manifest["folds"][7]["validation_subjects"].remove(39)
    manifest["validation_subjects_per_fold"] = 4
    path = _write(tmp_path, manifest)

    with pytest.raises(SplitManifestError, match="unassigned or out of range"):
        SplitManifest.load(path)


def test_subject_validating_in_two_folds_is_rejected(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_raw())
    manifest["folds"][1]["validation_subjects"] = [1, 12, 22, 32]  # 1 also validates in fold 0
    path = _write(tmp_path, manifest)

    with pytest.raises(SplitManifestError, match="more than one fold"):
        SplitManifest.load(path)


def test_fold_validating_on_a_test_subject_is_rejected(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_raw())
    manifest["folds"][0]["validation_subjects"] = [1, 11, 21, 5]  # 5 is held out for test
    path = _write(tmp_path, manifest)

    with pytest.raises(SplitManifestError, match="non-training subject"):
        SplitManifest.load(path)


def test_truncated_fold_list_is_rejected(tmp_path: Path) -> None:
    """A recomputed digest still cannot hide a structurally wrong manifest."""
    manifest = copy.deepcopy(_raw())
    manifest["folds"] = manifest["folds"][:7]
    path = _write(tmp_path, manifest)

    with pytest.raises(SplitManifestError, match="num_folds disagrees"):
        SplitManifest.load(path)


def test_overlapping_calibration_and_evaluation_repetitions_are_rejected(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_raw())
    manifest["repetitions"]["calibration"] = [1, 2]  # 2 is an evaluation repetition
    path = _write(tmp_path, manifest)

    with pytest.raises(SplitManifestError, match="leakage rule 3"):
        SplitManifest.load(path)


def test_missing_hashed_field_is_rejected(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_raw())
    manifest["manifest_hash"] = compute_hash(manifest)
    del manifest["repetitions"]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises((SplitManifestError, KeyError)):
        SplitManifest.load(path)


# --- the digest itself ------------------------------------------------------------------


def test_digest_ignores_formatting_and_comments() -> None:
    """Reflowing prose must not change the split's identity."""
    manifest = copy.deepcopy(_raw())
    manifest["notes"] = "a field outside HASHED_FIELDS"
    assert compute_hash(manifest) == compute_hash(_raw())


def test_digest_changes_when_any_subject_moves() -> None:
    manifest = copy.deepcopy(_raw())
    manifest["folds"][0]["validation_subjects"] = [11, 1, 21, 31]  # same set, different order
    assert compute_hash(manifest) != compute_hash(_raw()), "fold order is part of the split"

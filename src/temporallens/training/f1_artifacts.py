"""Verify an F1 artifact against what its consumer expects, before the consumer reads a weight.

:mod:`temporallens.training.subject_cv` writes provenance into every F1 artifact's
``model_config``. Writing it enforces nothing: the boundary only holds if each consumer checks it
(handoff §6, items 2–3; D47, D48, D76/D85). This module is that check, shared by every development
consumer — G2/G3/G4's fold encoders, the language arm's exact-B parents and refit — so each one
does not grow its own partial version.

**Expectations come from the consumer, never from the file.** A self-consistent file proves
nothing: a ``best.pt`` copied over ``fixed_budget.pt`` is internally consistent, and so is a refit
offered as a fold encoder. The caller states, in an :class:`F1Expectation`, the role it needs, the
fold, the accepted run's budget and horizon (from its own record of that run — the run summary or
a campaign manifest), the lineage it requires, the split manifest, and optionally the SHA-256 it
references the parent by (D76). The verifier compares the file with those, and recomputes what can
be recomputed from the permitted fitting pool — the fold's 28 training subjects, or all 32 for the
refit — with the same :func:`~temporallens.training.subject_cv.fitting_statistics` the trainer used:

* **Class weights (D85)**, hashed under the one versioned encoding; a missing or different digest
  is refused before any language fitting.
* **Source-window identity**, separately from the weight digest, because identical weights do not
  prove identical windows.
* **Normalization**, compared within a relative ``1e-6``. The statistics are accumulated in float64
  and stored as float32, so a recomputation on the same machine is bitwise equal; across machines
  a float64 summation-order difference can at most flip one float32 rounding (about ``1.2e-7``).
  Measured on real subjects 1–3 (pool {1, 2} against {1, 3} and against {1, 2, 3}), swapping or
  adding one subject changed the most-affected channel's standard deviation by 24–61% and its mean,
  which sits near zero, by over 1000%; the tolerance absorbs rounding and never a wrong pool.

Every mismatch is collected and reported together, so one failed verification names everything
wrong with the file rather than the first thing.

**Separate from the generic loader.** :mod:`temporallens.models.checkpoint` stays
protocol-agnostic, so F0's smoke checkpoint, which carries no provenance, still loads. A consumer
that needs an F1 parent calls this module; one that only needs a model calls the loader.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn

from temporallens.data.splits import SplitManifest
from temporallens.models.checkpoint import (
    CheckpointError,
    checkpoint_sha256,
    development_checkpoint_path,
    fold_checkpoint_path,
    load_checkpoint,
    rebuild_model,
    refit_checkpoint_path,
)
from temporallens.preprocessing.normalize import ChannelStats
from temporallens.training.class_weights import (
    CLASS_WEIGHTS_DIGEST_ENCODING,
    class_weights_sha256,
)
from temporallens.training.selection import (
    ESCALATED_HORIZON_EPOCHS,
    INITIAL_HORIZON_EPOCHS,
    SELECTION_WINDOW_EPOCHS,
    common_budget,
)
from temporallens.training.subject_cv import (
    ALL_32_REFIT,
    CANONICAL,
    COMMON_BUDGET_DEVELOPMENT,
    PRELIMINARY,
    SELECTED_FOLD_ANALYSIS,
    DevelopmentPool,
    FittingStatistics,
    fitting_statistics,
    lineage_root,
    window_identity_sha256,
)
from temporallens.utils.run_logger import config_digest

ROLES = (SELECTED_FOLD_ANALYSIS, COMMON_BUDGET_DEVELOPMENT, ALL_32_REFIT)
FOLD_ROLES = (SELECTED_FOLD_ANALYSIS, COMMON_BUDGET_DEVELOPMENT)
#: See the module docstring: one float32 rounding is ~1.2e-7; a one-subject pool change was >= 0.24.
NORMALIZATION_RELATIVE_TOLERANCE = 1e-6

#: Provenance every F1 artifact must carry. A superset of the F1 config's
#: ``training.development_checkpoints.required_source_provenance``, which the tests check.
_SHARED_PROVENANCE = (
    "role",
    "source_run_id",
    "accepted_horizon",
    "common_budget",
    "manifest_hash",
    "training_subject_ids",
    "normalization_sha256",
    "class_weights_sha256",
    "class_weights_digest_encoding",
    "source_window_identity_sha256",
    "config_digest",
    "training_rng_provenance",
    "device",
    "lineage",
    "git_commit",
    "git_dirty",
)
REQUIRED_PROVENANCE: dict[str, tuple[str, ...]] = {
    SELECTED_FOLD_ANALYSIS: _SHARED_PROVENANCE
    + ("fold_index", "scoring_subject_ids", "source_epoch", "source_checkpoint_sha256"),
    COMMON_BUDGET_DEVELOPMENT: _SHARED_PROVENANCE
    + (
        "fold_index",
        "scoring_subject_ids",
        "source_epoch",
        "training_budget",
        "source_checkpoint_sha256",
    ),
    ALL_32_REFIT: _SHARED_PROVENANCE + ("training_budget",),
}


class F1ArtifactError(RuntimeError):
    """An F1 artifact does not match what its consumer independently expects."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class F1Expectation:
    """What a consumer requires of one F1 artifact, from sources other than that artifact.

    ``fold`` is required for the two fold roles and forbidden for the refit. ``selected_epoch`` is
    the fold's own D18 epoch and is required for ``best.pt`` only: that artifact keeps its
    individually selected epoch, while ``budget`` is always the campaign's common B.
    """

    role: str
    budget: int
    horizon: int
    lineage: str
    manifest: SplitManifest
    fold: int | None = None
    selected_epoch: int | None = None
    sha256: str | None = None
    source_run_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown F1 artifact role {self.role!r}; expected one of {ROLES}")
        if self.horizon not in (INITIAL_HORIZON_EPOCHS, ESCALATED_HORIZON_EPOCHS) or not _is_int(
            self.horizon
        ):
            raise ValueError(f"an accepted F1 horizon is 30 or 60, got {self.horizon!r}")
        # D19: an accepted budget is eligible and strictly below its horizon.
        if not _is_int(self.budget) or not SELECTION_WINDOW_EPOCHS <= self.budget < self.horizon:
            raise ValueError(
                f"budget {self.budget!r} cannot be accepted at horizon {self.horizon} (D18/D19)"
            )
        if self.lineage not in (CANONICAL, PRELIMINARY):
            raise ValueError(f"unknown lineage {self.lineage!r}")
        if self.role in FOLD_ROLES:
            folds = self.manifest.num_folds
            if self.fold is None or not _is_int(self.fold) or not 0 <= self.fold < folds:
                raise ValueError(f"{self.role} needs a fold in [0, {folds}), got {self.fold!r}")
        elif self.fold is not None:
            raise ValueError("the all-32 refit belongs to no fold")
        if self.role == SELECTED_FOLD_ANALYSIS:
            epoch = self.selected_epoch
            if (
                epoch is None
                or not _is_int(epoch)
                or not SELECTION_WINDOW_EPOCHS <= epoch <= self.horizon
            ):
                raise ValueError(f"best.pt needs its own selected epoch, got {epoch!r}")
        elif self.selected_epoch is not None:
            raise ValueError("only best.pt has an individually selected epoch")
        if self.sha256 is not None and not _is_sha256(self.sha256):
            raise ValueError("sha256 must be a lowercase 64-hex digest")

    @property
    def training_subjects(self) -> list[int]:
        """The permitted fitting pool: the fold's training complement, or all 32 development."""
        if self.fold is None:
            return list(self.manifest.training_subjects)
        return list(self.manifest.training_subjects_for_fold(self.fold))

    @property
    def scoring_subjects(self) -> list[int]:
        """The fold's four validation subjects; the refit scores none during development."""
        return [] if self.fold is None else list(self.manifest.validation_subjects(self.fold))


@dataclass(frozen=True)
class VerifiedF1Artifact:
    """An F1 artifact that passed :func:`verify_f1_artifact`, with its rebuilt model."""

    path: Path
    sha256: str
    model: nn.Module
    model_config: dict[str, Any]

    @property
    def provenance(self) -> dict[str, Any]:
        """``model_config.provenance``, already checked against the expectation."""
        return dict(self.model_config["provenance"])

    @property
    def normalization(self) -> ChannelStats:
        """The statistics this model was trained under, already recomputed and compared."""
        return ChannelStats.from_dict(self.model_config["normalization"])

    @property
    def temperature(self) -> float | None:
        """The model's own frozen T, or ``None`` when it carries none or its fit failed."""
        calibration = self.model_config.get("probability_calibration") or {}
        if calibration.get("status") != "fitted":
            return None
        return float(calibration["temperature"])


def _compare(problems: list[str], field: str, recorded: Any, expected: Any) -> None:
    if type(recorded) is not type(expected) or recorded != expected:
        problems.append(f"{field} is {recorded!r}, expected {expected!r}")


def _check_normalization(
    problems: list[str], recorded: ChannelStats, recomputed: ChannelStats
) -> None:
    for name in ("mean", "std"):
        stored, fresh = getattr(recorded, name), getattr(recomputed, name)
        if stored.shape != fresh.shape or not np.allclose(
            stored, fresh, rtol=NORMALIZATION_RELATIVE_TOLERANCE, atol=0.0
        ):
            problems.append(
                f"normalization {name} does not match statistics refitted on the permitted pool"
            )
    _compare(problems, "normalization.num_samples", recorded.num_samples, recomputed.num_samples)
    _compare(problems, "normalization.fitted_on", recorded.fitted_on, recomputed.fitted_on)
    _compare(
        problems,
        "normalization.floored_channels",
        tuple(recorded.floored_channels),
        tuple(recomputed.floored_channels),
    )


def _check_calibration(
    problems: list[str], expected: F1Expectation, calibration: Mapping[str, Any] | None
) -> None:
    if expected.role == COMMON_BUDGET_DEVELOPMENT:
        # D84's rule, applied to F1 too: a retained exact-B snapshot is never mutated to carry T.
        if calibration is not None:
            problems.append("a common-budget development model must carry no temperature")
        return
    if not isinstance(calibration, Mapping):
        problems.append("probability_calibration is missing")
        return
    if expected.role == ALL_32_REFIT:
        _compare(problems, "calibration.status", calibration.get("status"), "fitted")
        _compare(
            problems,
            "calibration.role",
            calibration.get("role"),
            "refit_population_temperature",
        )
        _compare(
            problems,
            "calibration.prediction_source",
            calibration.get("prediction_source"),
            "pooled_subject_out_of_fold_fixed_budget_models",
        )
        _compare(
            problems,
            "calibration.subject_ids",
            calibration.get("subject_ids"),
            sorted(expected.training_subjects),
        )
    else:
        if calibration.get("status") not in ("fitted", "fit_failed"):
            problems.append(f"calibration.status is {calibration.get('status')!r}")
        if calibration.get("status") == "fitted":
            _compare(
                problems,
                "calibration.role",
                calibration.get("role"),
                "selected_fold_extended_analysis_only",
            )
            _compare(
                problems,
                "calibration.subject_ids",
                calibration.get("subject_ids"),
                sorted(expected.scoring_subjects),
            )
    if calibration.get("status") == "fitted":
        temperature = calibration.get("temperature")
        if not isinstance(temperature, float) or not (
            math.isfinite(temperature) and temperature > 0
        ):
            problems.append(f"calibration.temperature {temperature!r} is not finite and positive")
        _compare(
            problems,
            "calibration.excluded_repetitions",
            calibration.get("excluded_repetitions"),
            [0],
        )


def _check_provenance(
    problems: list[str], expected: F1Expectation, provenance: Mapping[str, Any]
) -> None:
    missing = [key for key in REQUIRED_PROVENANCE[expected.role] if key not in provenance]
    if missing:
        problems.append(f"provenance is missing {missing}")
    get = provenance.get
    _compare(problems, "role", get("role"), expected.role)
    _compare(problems, "accepted_horizon", get("accepted_horizon"), expected.horizon)
    _compare(problems, "common_budget", get("common_budget"), expected.budget)
    _compare(problems, "manifest_hash", get("manifest_hash"), expected.manifest.manifest_hash)
    _compare(problems, "lineage", get("lineage"), expected.lineage)
    _compare(
        problems,
        "class_weights_digest_encoding",
        get("class_weights_digest_encoding"),
        CLASS_WEIGHTS_DIGEST_ENCODING,
    )
    _compare(
        problems, "training_subject_ids", get("training_subject_ids"), expected.training_subjects
    )
    if expected.source_run_id is not None:
        _compare(problems, "source_run_id", get("source_run_id"), expected.source_run_id)

    if expected.role in FOLD_ROLES:
        _compare(problems, "fold_index", get("fold_index"), expected.fold)
        _compare(
            problems, "scoring_subject_ids", get("scoring_subject_ids"), expected.scoring_subjects
        )
        if not _is_sha256(get("source_checkpoint_sha256")):
            problems.append("source_checkpoint_sha256 is not a SHA-256 digest")
    else:
        for key in ("fold_index", "scoring_subject_ids", "source_epoch"):
            if key in provenance:
                problems.append(f"an all-32 refit must not record {key}")
    if expected.role == SELECTED_FOLD_ANALYSIS:
        _compare(problems, "source_epoch", get("source_epoch"), expected.selected_epoch)
    elif expected.role == COMMON_BUDGET_DEVELOPMENT:
        _compare(problems, "source_epoch", get("source_epoch"), expected.budget)
        _compare(problems, "training_budget", get("training_budget"), expected.budget)
    else:
        _compare(problems, "training_budget", get("training_budget"), expected.budget)

    held_out = set(expected.manifest.test_subjects)
    named: set[Any] = set()
    for key in ("training_subject_ids", "scoring_subject_ids"):
        value = get(key)
        if isinstance(value, list):  # a malformed field is already reported above
            named.update(value)
    if named & held_out:
        problems.append(f"names final-test subject(s) {sorted(named & held_out)}")
    if expected.lineage == CANONICAL:
        # D47: a canonical parent came from committed code on CUDA, as recorded at training time.
        if not str(get("device", "")).startswith("cuda"):
            problems.append(
                f"a canonical artifact must record a CUDA device, not {get('device')!r}"
            )
        if get("git_dirty") is not False:
            problems.append("a canonical artifact must come from a clean working tree")


def verify_f1_artifact(
    path: str | Path, expected: F1Expectation, *, pool: DevelopmentPool
) -> VerifiedF1Artifact:
    """Load ``path`` only if it is the F1 artifact ``expected`` describes; raise otherwise.

    ``pool`` must hold the development recordings indexed with F1's window size and stride; a
    different windowing is caught by the source-window identity. Raises :class:`F1ArtifactError`
    listing every mismatch, including a missing file and a file the generic loader refuses.
    """
    return _verify(path, expected, fitting_statistics(pool, expected.training_subjects))


def _verify(
    path: str | Path, expected: F1Expectation, statistics: FittingStatistics
) -> VerifiedF1Artifact:
    # ``statistics`` is always recomputed by this module for expected.training_subjects; callers
    # outside it cannot supply their own, or the recomputation would stop being independent.
    source = Path(path)
    if not source.is_file():
        raise F1ArtifactError(f"{source}: missing F1 artifact ({expected.role})")
    digest = checkpoint_sha256(source)
    problems: list[str] = []
    if expected.sha256 is not None and digest != expected.sha256:
        problems.append(f"file SHA-256 is {digest}, the consumer references {expected.sha256}")
    try:
        _, model_config = load_checkpoint(source)
        model = rebuild_model(source)
    except CheckpointError as error:
        raise F1ArtifactError(f"{source}: {error}") from error
    if checkpoint_sha256(source) != digest:
        raise F1ArtifactError(f"{source}: changed while it was being verified")

    provenance = model_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise F1ArtifactError(f"{source}: carries no provenance, so it is not an F1 artifact")
    _check_provenance(problems, expected, provenance)
    _check_calibration(problems, expected, model_config.get("probability_calibration"))

    # Recomputed from the permitted pool, never from the file's own subject list.
    _compare(
        problems,
        "class_weights_sha256",
        provenance.get("class_weights_sha256"),
        class_weights_sha256(statistics.class_weights),
    )
    _compare(
        problems,
        "source_window_identity_sha256",
        provenance.get("source_window_identity_sha256"),
        window_identity_sha256(statistics.index),
    )
    recorded = ChannelStats.from_dict(model_config["normalization"])
    _compare(
        problems,
        "normalization_sha256",
        provenance.get("normalization_sha256"),
        config_digest(model_config["normalization"]),
    )
    _check_normalization(problems, recorded, statistics.normalization)

    if problems:
        raise F1ArtifactError(
            f"{source} is not the expected {expected.role} artifact:\n- " + "\n- ".join(problems)
        )
    return VerifiedF1Artifact(path=source, sha256=digest, model=model, model_config=model_config)


@dataclass(frozen=True)
class VerifiedF1Run:
    """All seventeen artifacts of one accepted F1 run, verified individually and together."""

    selected: tuple[VerifiedF1Artifact, ...]
    development: tuple[VerifiedF1Artifact, ...]
    refit: VerifiedF1Artifact


def verify_f1_run(
    checkpoint_base: str | Path,
    experiment: str,
    *,
    selected_epochs: Sequence[int],
    horizon: int,
    lineage: str,
    manifest: SplitManifest,
    pool: DevelopmentPool,
    source_run_id: str | None = None,
) -> VerifiedF1Run:
    """Verify every artifact of one F1 run and the links between them.

    ``selected_epochs`` are the eight D18 epochs from the consumer's record of the run; the common
    budget is recomputed from them by D18's rule rather than read from any artifact. Beyond each
    artifact's own checks, all seventeen must share one source run, and the refit's temperature
    must name exactly these eight development models, by digest and source snapshot, in fold order.
    """
    epochs = list(selected_epochs)
    budget = common_budget(epochs, expected_folds=manifest.num_folds)
    root = lineage_root(checkpoint_base, lineage)
    run_id = source_run_id
    # best.pt and fixed_budget.pt share a fold's pool: fit its statistics once, not twice.
    fitted: dict[tuple[int, ...], FittingStatistics] = {}

    def check(path: Path, **fields: Any) -> VerifiedF1Artifact:
        nonlocal run_id
        expected = F1Expectation(
            budget=budget,
            horizon=horizon,
            lineage=lineage,
            manifest=manifest,
            source_run_id=run_id,
            **fields,
        )
        key = tuple(expected.training_subjects)
        if key not in fitted:
            fitted[key] = fitting_statistics(pool, expected.training_subjects)
        verified = _verify(path, expected, fitted[key])
        run_id = verified.provenance["source_run_id"]  # the first artifact fixes the run
        return verified

    selected = tuple(
        check(
            fold_checkpoint_path(root, experiment, fold=k),
            role=SELECTED_FOLD_ANALYSIS,
            fold=k,
            selected_epoch=epochs[k],
        )
        for k in range(manifest.num_folds)
    )
    development = tuple(
        check(
            development_checkpoint_path(root, experiment, fold=k),
            role=COMMON_BUDGET_DEVELOPMENT,
            fold=k,
        )
        for k in range(manifest.num_folds)
    )
    refit = check(refit_checkpoint_path(root, experiment), role=ALL_32_REFIT)

    calibration = refit.model_config["probability_calibration"]
    problems: list[str] = []
    _compare(
        problems,
        "refit calibration.fold_artifact_sha256",
        calibration.get("fold_artifact_sha256"),
        [d.sha256 for d in development],
    )
    trajectory = calibration.get("source_trajectory_provenance")
    expected_trajectory = [
        {
            "fold_index": k,
            "source_run_id": run_id,
            "source_epoch": budget,
            "source_checkpoint_sha256": d.provenance["source_checkpoint_sha256"],
            "accepted_horizon": horizon,
        }
        for k, d in enumerate(development)
    ]
    _compare(
        problems, "refit calibration.source_trajectory_provenance", trajectory, expected_trajectory
    )
    if problems:
        raise F1ArtifactError(
            f"{refit.path}: its temperature is not from this run's development models:\n- "
            + "\n- ".join(problems)
        )
    return VerifiedF1Run(selected=selected, development=development, refit=refit)

"""F1's development harness: eight subject folds, D18–D21 selection, D48 artifacts, D33 T, refit.

This is the procedure that turns the frozen split into the encoder both arms load. It never loads
a final-test subject: the eight held-out subjects are scored only by a separate, CUDA-only stage
(D47), so nothing here can leak them into a fit or a selection.

The order of operations is the protocol:

1. **Eight selection trajectories** at the 30-epoch horizon (D18), each fitted on its fold's 28
   training subjects with normalization and class weights from those subjects only (§3.3, D11),
   validated every epoch on its four validation subjects with subject-equal macro-F1 (D21).
2. **Selection and the horizon rule.** Each fold's D18 epoch, then the common budget B; a budget at
   30 reruns all eight folds from scratch at 60, and a second boundary hit stops (D19). No refit
   and no final-test prediction exists before this resolves.
3. **Development artifacts (D48).** Every eligible epoch of every trajectory is snapshotted to disk
   until B is known. ``folds/fold{k}/best.pt`` is the fold's own selected epoch (analysis only),
   ``development/fold{k}/fixed_budget.pt`` its trajectory at the shared B. Both copy parameters and
   buffers exactly, and both are verified against the snapshot digest recorded when it was taken.
4. **Temperatures.** The refit's one T comes from the pooled out-of-fold logits of the eight
   fixed-budget models (D33/D39), repetitions 1–6 only (D81). Each ``best.pt`` gets its own T from
   its four validation subjects for extended analysis, and D40's leave-one-subject-out fits give
   scaled validation diagnostics. None of these feeds another.
5. **The refit** trains a fresh model on all 32 development subjects for exactly B epochs, with
   statistics and weights refitted on those 32, and carries the pooled T.

**Validation cannot change training (D48).** A validation pass runs in eval mode under no-grad,
through a loader with its own generator, inside a block that restores every relevant RNG state;
the training loader has its own generator too. The first B epochs are therefore the same fitting
procedure whatever the horizon or validation results, which is what makes reusing a trajectory's
epoch-B snapshot equivalent to a fixed-budget fit. The harness tests this rather than assuming it.

**D38's recipe.** ``torch.manual_seed(seed)`` before building each model (PyTorch-default
initialization under ``experiment.seed``), a training-order generator seeded with the same seed,
FP32, Adam with betas (0.9, 0.999), epsilon 1e-8, no AMSGrad, coupled L2 on every parameter, a
constant learning rate, no clipping or mixed precision. A non-finite loss or gradient raises before
the optimizer step, never a silently skipped update.

**Execution scope and lineage (D47).** A run's scope is requested explicitly; it is never inferred
from the device or from a clean tree, because neither proves the procedure is complete. A
preliminary run writes under ``<base>/preliminary/`` on every device, CUDA included, so it can
never produce a file at a path a reportable consumer reads. A reportable run would write the
canonical paths, and it is refused while :data:`REPORTABLE_EXECUTION_BLOCKERS` is non-empty: the
final-test stage is not written, and has not been rehearsed on development stand-ins. The
clean-tree guard is separate and applies to both scopes. Every artifact records its lineage and
device, and :mod:`temporallens.training.f1_artifacts` checks both at consumption.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader

from temporallens.data.dataset import WindowDataset
from temporallens.data.ninapro import (
    NUM_CHANNELS,
    NUM_CLASSES,
    SubjectRecording,
    load_processed,
    processed_path,
)
from temporallens.data.splits import SplitManifest
from temporallens.data.windows import WindowIndex, concatenate, index_recording
from temporallens.evaluation.calibration import (
    EXCLUDED_FITTING_REPETITIONS,
    TemperatureFit,
    TemperatureFitError,
    cross_fitted_temperatures,
    d33_fitting_weights,
    fit_population_temperature,
    scaled_probabilities,
)
from temporallens.evaluation.metrics import (
    accuracy,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    macro_f1,
    overconfidence_error,
)
from temporallens.models.checkpoint import (
    checkpoint_sha256,
    development_checkpoint_path,
    fold_checkpoint_path,
    refit_checkpoint_path,
    save_checkpoint,
)
from temporallens.models.encoders.cnn1d import MODEL_TYPE, build_model
from temporallens.preprocessing.normalize import ChannelStats, fit_channel_stats
from temporallens.training.class_weights import (
    CLASS_WEIGHTS_DIGEST_ENCODING,
    class_weights_sha256,
    inverse_frequency_weights,
)
from temporallens.training.selection import (
    INITIAL_HORIZON_EPOCHS,
    SELECTION_WINDOW_EPOCHS,
    EpochSelection,
    HorizonDecision,
    common_budget,
    decide_horizon,
    fold_score,
    select_epoch,
)
from temporallens.utils.device import resolve_device
from temporallens.utils.run_logger import RunLogger, config_digest

#: D38's Adam mechanics; the learning rate and decay come from the config.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
#: D47's artifact lineages. Only a reportable run writes the canonical paths consumers read.
CANONICAL = "canonical"
PRELIMINARY = "preliminary"
#: Execution scopes. A preliminary run is the lineage of the same name; a reportable one is canonical.
REPORTABLE = "reportable"
#: Why a reportable run is refused. Remove an entry only in the change that implements it.
REPORTABLE_EXECUTION_BLOCKERS: tuple[str, ...] = (
    "the CUDA-only final-test stage (D6, D8, D10/D83, D80) is not implemented",
    "that stage has not been rehearsed on development-subject stand-ins (handoff §6, item 5)",
)
#: The refit's fit index, after the eight folds, in logs and snapshot paths.
REFIT = "refit"
#: The three D48 artifact roles, as recorded in ``model_config.provenance.role``.
SELECTED_FOLD_ANALYSIS = "selected_fold_analysis"
COMMON_BUDGET_DEVELOPMENT = "common_budget_development"
ALL_32_REFIT = "all_32_refit"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class NonFiniteTrainingError(RuntimeError):
    """A loss or gradient was not finite; D38 fails the fit rather than skipping the update."""


class HorizonBlockedError(RuntimeError):
    """D19's second boundary hit: the common budget equals the escalated horizon."""


class SnapshotError(RuntimeError):
    """A retained epoch snapshot is missing or no longer matches the digest taken when saved."""


class ArtifactExistsError(RuntimeError):
    """A frozen artifact path is already occupied; F1 artifacts are written once."""


class ReportableExecutionError(RuntimeError):
    """A reportable F1 run was requested that the harness cannot honestly perform yet."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class SubjectCVConfig:
    """Everything the F1 harness needs, with the split supplied as a verified manifest."""

    name: str
    seed: int
    data_dir: Path
    manifest: SplitManifest
    window_size: int
    stride: int
    hidden_dim: int
    embedding_dim: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str = "auto"
    num_workers: int = 0
    input_channels: int = NUM_CHANNELS
    num_classes: int = NUM_CLASSES

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if not _is_int(self.seed) or not 0 <= self.seed < 2**32:
            raise ValueError(f"seed must be an integer in [0, 2**32), got {self.seed!r}")
        for name in ("window_size", "stride", "hidden_dim", "embedding_dim", "batch_size"):
            value = getattr(self, name)
            if not _is_int(value) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.stride > self.window_size:
            raise ValueError("stride must not exceed window_size")
        if self.input_channels != NUM_CHANNELS or self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"DB2 Exercise B has {NUM_CHANNELS} channels and {NUM_CLASSES} classes"
            )
        for name in ("learning_rate", "weight_decay", "dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.learning_rate <= 0 or self.dropout >= 1:
            raise ValueError("learning_rate must be positive and dropout below 1")
        if not _is_int(self.num_workers) or self.num_workers < 0:
            raise ValueError(f"num_workers must be a non-negative integer, got {self.num_workers}")

    def model_spec(self) -> dict[str, Any]:
        """The ``build_model`` arguments every fit in this run uses."""
        return {
            "type": MODEL_TYPE,
            "input_channels": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }

    def loggable(self) -> dict[str, Any]:
        """JSON-safe description, with the manifest identified by its digest."""
        out: dict[str, Any] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(self).items()
            if key != "manifest"
        }
        out["manifest_hash"] = self.manifest.manifest_hash
        return out


def execution_lineage(scope: str, device: torch.device, *, clean_tree_required: bool) -> str:
    """D47: the lineage a run's artifacts take from its explicitly requested ``scope``.

    A preliminary scope is preliminary on every device. A reportable scope is refused while any
    :data:`REPORTABLE_EXECUTION_BLOCKERS` remains, and otherwise only on CUDA from a run that
    requires a clean tree; it is never granted because the device happens to be CUDA.
    """
    if scope == PRELIMINARY:
        return PRELIMINARY
    if scope != REPORTABLE:
        raise ValueError(
            f"unknown execution scope {scope!r}; use {PRELIMINARY!r} or {REPORTABLE!r}"
        )
    if REPORTABLE_EXECUTION_BLOCKERS:
        raise ReportableExecutionError(
            "refusing a reportable F1 run: " + "; ".join(REPORTABLE_EXECUTION_BLOCKERS)
        )
    if device.type != "cuda":
        raise ReportableExecutionError(f"a reportable F1 run is CUDA-only (D47), not {device}")
    if not clean_tree_required:
        raise ReportableExecutionError("a reportable F1 run must require a clean working tree")
    return CANONICAL


def lineage_root(base: str | Path, lineage: str) -> Path:
    """The checkpoint root for a lineage; preliminary artifacts never share canonical paths."""
    if lineage not in (CANONICAL, PRELIMINARY):
        raise ValueError(f"unknown lineage {lineage!r}")
    return Path(base) if lineage == CANONICAL else Path(base) / PRELIMINARY


# --- data -------------------------------------------------------------------------------------


@dataclass
class DevelopmentPool:
    """The 32 development recordings and their window indices, loaded once per run."""

    recordings: dict[int, SubjectRecording]
    indices: dict[int, WindowIndex]

    def index(self, subjects: Sequence[int]) -> WindowIndex:
        """The concatenated windows of ``subjects``, in the order given."""
        return concatenate([self.indices[s] for s in subjects])

    def recordings_for(self, subjects: Sequence[int]) -> dict[int, SubjectRecording]:
        """The recordings of ``subjects`` only, so a fitting call cannot reach any other."""
        return {s: self.recordings[s] for s in subjects}


def load_development_pool(config: SubjectCVConfig) -> DevelopmentPool:
    """Load the manifest's development subjects — never a final-test subject (D16, D47)."""
    subjects = list(config.manifest.training_subjects)
    if set(subjects) & set(config.manifest.test_subjects):
        raise ValueError("the manifest lists a subject as both development and final test")
    recordings = {s: load_processed(processed_path(config.data_dir, s)) for s in subjects}
    indices = {
        s: index_recording(r, window_size=config.window_size, stride=config.stride)
        for s, r in recordings.items()
    }
    return DevelopmentPool(recordings=recordings, indices=indices)


@dataclass(frozen=True)
class FittingStatistics:
    """What a fit derives from its training subjects alone: windows, statistics, class weights."""

    index: WindowIndex
    normalization: ChannelStats
    class_weights: npt.NDArray[np.float32]


def fitting_statistics(
    pool: DevelopmentPool, train_subjects: Sequence[int], *, num_classes: int = NUM_CLASSES
) -> FittingStatistics:
    """Normalization and D11 class weights fitted on ``train_subjects`` only (§3.3).

    The one definition the trainer uses to fit them and the F1 artifact verifier uses to
    recompute them, so a consumer checks a recorded digest against the permitted pool rather
    than against the producer's own claim.
    """
    subjects = list(train_subjects)
    index = pool.index(subjects)
    normalization = fit_channel_stats(
        pool.recordings_for(subjects), index, fitted_on=f"train_subjects={sorted(subjects)}"
    )
    class_weights = inverse_frequency_weights(index.label, num_classes=num_classes)
    return FittingStatistics(index=index, normalization=normalization, class_weights=class_weights)


def window_identity_sha256(index: WindowIndex) -> str:
    """Digest of exactly which windows a fit used, kept separate from the weight digest (D85)."""
    digest = hashlib.sha256()
    for name, values, dtype in (
        ("subject", index.subject, "<i2"),
        ("start", index.start, "<i8"),
        ("label", index.label, "<i8"),
        ("repetition", index.repetition, "<i2"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(f"{name}:{dtype}:{list(array.shape)}:{index.window_size}\n".encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


# --- one fit ----------------------------------------------------------------------------------


@contextlib.contextmanager
def preserved_rng(device: torch.device) -> Iterator[None]:
    """Restore the CPU and device RNG states on exit, so the enclosed work cannot advance them."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    mps = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


@dataclass(frozen=True)
class Predictions:
    """One model's logits on a set of windows, with each window's subject and repetition."""

    logits: FloatArray
    labels: IntArray
    subjects: IntArray
    repetitions: IntArray

    def take(self, mask: npt.NDArray[np.bool_]) -> Predictions:
        """The windows selected by ``mask``."""
        return Predictions(
            self.logits[mask], self.labels[mask], self.subjects[mask], self.repetitions[mask]
        )

    def logits_sha256(self) -> str:
        """Digest of the float64 logits and their window identities, for calibrator provenance."""
        digest = hashlib.sha256()
        for name, values, dtype in (
            ("logits", self.logits, "<f8"),
            ("label", self.labels, "<i8"),
            ("subject", self.subjects, "<i8"),
            ("repetition", self.repetitions, "<i8"),
        ):
            array = np.ascontiguousarray(values, dtype=dtype)
            digest.update(f"{name}:{dtype}:{list(array.shape)}\n".encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()


def predict(
    model: nn.Module,
    index: WindowIndex,
    recordings: Mapping[int, SubjectRecording],
    normalization: ChannelStats,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> Predictions:
    """Logits for every window of ``index``, in order, without touching any training RNG."""
    loader = DataLoader(
        WindowDataset(index, recordings, normalization=normalization),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # Iterator creation draws a base seed; its own generator keeps that off the global stream.
        generator=torch.Generator().manual_seed(0),
    )
    was_training = model.training
    model.eval()
    chunks: list[np.ndarray] = []
    with preserved_rng(device), torch.no_grad():
        for windows, _, _ in loader:
            chunks.append(model(windows.to(device)).detach().cpu().numpy().astype(np.float64))
    model.train(was_training)
    return Predictions(
        logits=np.concatenate(chunks),
        labels=index.label.astype(np.int64),
        subjects=index.subject.astype(np.int64),
        repetitions=index.repetition.astype(np.int64),
    )


def subject_macro_f1(
    predictions: Predictions, *, num_classes: int = NUM_CLASSES
) -> dict[int, float]:
    """18-class macro-F1 per subject, undefined classes counted as 0 (D21)."""
    predicted = predictions.logits.argmax(axis=1)
    return {
        int(s): macro_f1(
            predicted[predictions.subjects == s],
            predictions.labels[predictions.subjects == s],
            num_classes=num_classes,
        )
        for s in np.unique(predictions.subjects)
    }


@dataclass
class FitResult:
    """One completed fit: its final model, fitting statistics, curve and retained snapshots."""

    model: nn.Module
    normalization: ChannelStats
    class_weights: npt.NDArray[np.float32]
    epochs: int
    epoch_losses: list[float]
    raw_scores: list[float]
    per_subject_scores: list[dict[int, float]]
    snapshots: dict[int, tuple[Path, str]]
    train_window_count: int
    window_identity_sha256: str
    rng_provenance: dict[str, Any]


def _rng_provenance(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "initialization": "torch_manual_seed_then_pytorch_default_module_init",
        "torch_manual_seed": seed,
        "training_order_generator_seed": seed,
        "validation_generator": "dedicated_and_rng_state_restored",
        "torch_version": str(torch.__version__),  # TorchVersion is not weights_only-safe
        "device": str(device),
    }


def _save_snapshot(model: nn.Module, path: Path) -> str:
    if path.exists():
        raise SnapshotError(f"refusing to overwrite retained snapshot {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state, path)
    return checkpoint_sha256(path)


def fit_population_model(
    config: SubjectCVConfig,
    pool: DevelopmentPool,
    *,
    train_subjects: Sequence[int],
    validation_subjects: Sequence[int] = (),
    epochs: int,
    snapshot_dir: Path | None = None,
    device: torch.device,
    on_epoch: Callable[[int, dict[str, Any]], None] | None = None,
) -> FitResult:
    """Fit one model under D38, validating and snapshotting each eligible epoch when asked.

    Normalization and class weights are fitted on ``train_subjects`` only. Validation subjects
    are scored after every epoch; epochs from :data:`SELECTION_WINDOW_EPOCHS` onward are
    snapshotted into ``snapshot_dir`` so D48 can read the trajectory at any eligible budget.
    """
    train_subjects, validation_subjects = list(train_subjects), list(validation_subjects)
    if set(train_subjects) & set(validation_subjects):
        raise ValueError("a subject cannot both train and validate the same fit")
    if not _is_int(epochs) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    statistics = fitting_statistics(pool, train_subjects, num_classes=config.num_classes)
    train_index = statistics.index
    train_recordings = pool.recordings_for(train_subjects)
    normalization = statistics.normalization
    class_weights = statistics.class_weights

    torch.manual_seed(config.seed)  # D38: PyTorch-default initialization under the seed
    model = build_model(config.model_spec()).to(device)
    weight_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    train_loader = DataLoader(
        WindowDataset(train_index, train_recordings, normalization=normalization),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_index = pool.index(validation_subjects) if validation_subjects else None
    validation_recordings = pool.recordings_for(validation_subjects)

    epoch_losses: list[float] = []
    raw_scores: list[float] = []
    per_subject_scores: list[dict[int, float]] = []
    snapshots: dict[int, tuple[Path, str]] = {}
    parameters = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(1, epochs + 1):  # fixed horizon, no early stopping (D18/D19)
        model.train()
        batches: list[tuple[float, float]] = []
        for windows, labels, _ in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(windows.to(device)), labels)
            if not bool(torch.isfinite(loss)):
                raise NonFiniteTrainingError(f"non-finite loss at epoch {epoch}; no update made")
            loss.backward()
            finite = torch.stack(
                [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
            )
            if not bool(finite.all()):
                raise NonFiniteTrainingError(
                    f"non-finite gradient at epoch {epoch}; no update made"
                )
            optimizer.step()
            batches.append((float(loss.detach()), float(weight_tensor[labels].sum())))
        total_weight = math.fsum(w for _, w in batches)
        epoch_losses.append(math.fsum(loss * w for loss, w in batches) / total_weight)

        record: dict[str, Any] = {"epoch": epoch, "train_loss": epoch_losses[-1]}
        if validation_index is not None:
            scores = subject_macro_f1(
                predict(
                    model,
                    validation_index,
                    validation_recordings,
                    normalization,
                    device=device,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                ),
                num_classes=config.num_classes,
            )
            per_subject_scores.append(scores)
            raw_scores.append(fold_score(scores, expected_subjects=len(validation_subjects)))
            record["validation_macro_f1_by_subject"] = {str(s): v for s, v in scores.items()}
            record["validation_fold_score"] = raw_scores[-1]
        if snapshot_dir is not None and epoch >= SELECTION_WINDOW_EPOCHS:
            path = snapshot_dir / f"epoch{epoch:03d}.pt"
            snapshots[epoch] = (path, _save_snapshot(model, path))
        if on_epoch is not None:
            on_epoch(epoch, record)

    return FitResult(
        model=model,
        normalization=normalization,
        class_weights=class_weights,
        epochs=epochs,
        epoch_losses=epoch_losses,
        raw_scores=raw_scores,
        per_subject_scores=per_subject_scores,
        snapshots=snapshots,
        train_window_count=len(train_index),
        window_identity_sha256=window_identity_sha256(train_index),
        rng_provenance=_rng_provenance(config.seed, device),
    )


def load_snapshot(config: SubjectCVConfig, fit: FitResult, epoch: int) -> nn.Module:
    """Rebuild the fit's model at ``epoch`` from its retained snapshot, verifying the digest."""
    if epoch not in fit.snapshots:
        raise SnapshotError(f"no retained snapshot for epoch {epoch}; refusing a substitute")
    path, expected = fit.snapshots[epoch]
    if not path.is_file() or checkpoint_sha256(path) != expected:
        raise SnapshotError(f"snapshot {path} is missing or changed since it was taken")
    model = build_model(config.model_spec())
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# --- the harness ------------------------------------------------------------------------------


#: The injection seam: tests replace the fitting function, never the protocol around it.
Trainer = Callable[..., FitResult]


@dataclass
class SubjectCVResult:
    """What the harness produced: the accepted horizon, budget, artifacts and calibration."""

    run_dir: Path
    lineage: str
    accepted_horizon: int
    budget: int
    selections: list[EpochSelection]
    decisions: list[HorizonDecision]
    artifacts: dict[str, Path]
    refit_temperature: TemperatureFit
    summary: dict[str, Any] = field(default_factory=dict)


def _artifact_paths(root: Path, name: str, folds: int) -> dict[str, Path]:
    paths = {"refit": refit_checkpoint_path(root, name)}
    for k in range(folds):
        paths[f"fold{k}_best"] = fold_checkpoint_path(root, name, fold=k)
        paths[f"fold{k}_fixed_budget"] = development_checkpoint_path(root, name, fold=k)
    return paths


def _calibration_metrics(predictions: Predictions, temperature: float | None) -> dict[str, Any]:
    """Raw or scaled calibration metrics, with bins recomputed for these probabilities (D23)."""
    probabilities = (
        scaled_probabilities(predictions.logits, temperature)
        if temperature is not None
        else scaled_probabilities(predictions.logits, 1.0)
    )
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == predictions.labels
    bins = calibration_bins(confidence, correct)
    return {
        "expected_calibration_error": expected_calibration_error(confidence, correct),
        "overconfidence_error": overconfidence_error(confidence, correct),
        "brier_score": brier_score(probabilities, predictions.labels),
        "effective_bin_count": bins.effective_count,
        "bin_occupancy": bins.occupancy.tolist(),
    }


def _validation_metrics(predictions: Predictions, num_classes: int) -> dict[str, Any]:
    """D20: raw validation metrics from the selected checkpoint, per subject then equal-mean."""
    per_subject: dict[str, dict[str, Any]] = {}
    for s in np.unique(predictions.subjects):
        own = predictions.take(predictions.subjects == s)
        predicted = own.logits.argmax(axis=1)
        per_subject[str(int(s))] = {
            "macro_f1": macro_f1(predicted, own.labels, num_classes=num_classes),
            "accuracy": accuracy(predicted, own.labels),
            **{f"{k}_raw": v for k, v in _calibration_metrics(own, None).items()},
        }
    keys = ("macro_f1", "accuracy", "expected_calibration_error_raw", "brier_score_raw")
    mean = {k: math.fsum(m[k] for m in per_subject.values()) / len(per_subject) for k in keys}
    return {"per_subject": per_subject, "subject_mean": mean}


def _fitting_weights_record(predictions: Predictions) -> dict[str, Any]:
    """The D33/D81 weights a fit on ``predictions`` uses: scheme, digest and per-cell totals.

    The totals make the equal weighting checkable from the record alone: each subject sums to
    1/n, its repetitions share that equally, and repetition 0 shows as an explicit 0.
    """
    weights = d33_fitting_weights(predictions.subjects, predictions.repetitions)
    totals: dict[str, dict[str, float]] = {}
    for s in np.unique(predictions.subjects):
        own = predictions.subjects == s
        totals[str(int(s))] = {
            str(int(r)): math.fsum(weights[own & (predictions.repetitions == r)].tolist())
            for r in np.unique(predictions.repetitions[own])
        }
    return {
        "fitting_weights": "normalized_d33_equal_subject_repetition_window",
        "fitting_weights_sha256": hashlib.sha256(
            np.ascontiguousarray(weights, dtype="<f8").tobytes()
        ).hexdigest(),
        "fitting_weight_by_subject_repetition": totals,
    }


def _cross_fitted_diagnostics(
    predictions: Predictions, *, model_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """D40: score each validation subject with the T fitted on the other three.

    ``model_identity`` names the one frozen model every fit used. Each fit persists its fitting
    and scored subjects, its weights and the digest of the logits it was fitted on. A failed fit
    is recorded as blocked, never replaced by T=1 (D39).
    """
    try:
        fits = cross_fitted_temperatures(
            predictions.logits, predictions.labels, predictions.subjects, predictions.repetitions
        )
    except TemperatureFitError as error:
        return {"status": "blocked", "reason": str(error), "model_identity": dict(model_identity)}
    per_subject: dict[str, dict[str, Any]] = {}
    for scored, fit in fits.items():
        own = predictions.take(predictions.subjects == scored)
        fitting = predictions.take(predictions.subjects != scored)
        per_subject[str(scored)] = {
            "scored_subject": int(scored),
            "fitting_subjects": sorted(int(s) for s in np.unique(fitting.subjects)),
            "fitting_logits_sha256": fitting.logits_sha256(),
            **_fitting_weights_record(fitting),
            "temperature": fit.temperature,
            "fit": fit.to_dict(),
            **{
                f"{k}_temperature_scaled": v
                for k, v in _calibration_metrics(own, fit.temperature).items()
            },
        }
    keys = ("expected_calibration_error_temperature_scaled", "brier_score_temperature_scaled")
    mean = {k: math.fsum(m[k] for m in per_subject.values()) / len(per_subject) for k in keys}
    return {
        "status": "fitted",
        "role": "development_diagnostic_conditional_on_selected_checkpoint",
        "model_identity": dict(model_identity),
        "per_subject": per_subject,
        "subject_mean": mean,
    }


def _calibration_record(
    fit: TemperatureFit, predictions: Predictions, *, source: str, **extra: Any
) -> dict[str, Any]:
    return {
        "method": "temperature_scaling",
        "status": "fitted",
        "temperature": fit.temperature,
        "prediction_source": source,
        "subject_ids": sorted(int(s) for s in np.unique(predictions.subjects)),
        "repetition_ids": sorted(
            int(r)
            for r in np.unique(predictions.repetitions)
            if r not in EXCLUDED_FITTING_REPETITIONS
        ),
        "excluded_repetitions": list(EXCLUDED_FITTING_REPETITIONS),
        "query_logits_sha256": predictions.logits_sha256(),
        **_fitting_weights_record(predictions),
        "fit_diagnostics": fit.to_dict(),
        **extra,
    }


def _concatenate(parts: Sequence[Predictions]) -> Predictions:
    return Predictions(
        logits=np.concatenate([p.logits for p in parts]),
        labels=np.concatenate([p.labels for p in parts]),
        subjects=np.concatenate([p.subjects for p in parts]),
        repetitions=np.concatenate([p.repetitions for p in parts]),
    )


def run_subject_cross_validation(
    config: SubjectCVConfig,
    *,
    checkpoint_base: str | Path = "checkpoints",
    output_dir: str | Path = "results/runs",
    metadata: Mapping[str, Any] | None = None,
    scope: str = PRELIMINARY,
    require_clean_tree: bool = False,
    prune_snapshots: bool = True,
    trainer: Trainer = fit_population_model,
    pool: DevelopmentPool | None = None,
) -> SubjectCVResult:
    """Run F1's development procedure end to end and write its artifacts once.

    ``scope`` decides the lineage (:func:`execution_lineage`); ``require_clean_tree`` is the
    separate guard that the code is committed. Raises :class:`ReportableExecutionError` for a
    reportable scope before anything is written, :class:`HorizonBlockedError` on D19's second
    boundary hit, after logging the curves and before any refit, and
    :class:`~temporallens.evaluation.calibration.TemperatureFitError` if the pooled temperature
    fit fails D39's checks, before the refit trains.
    """
    device = resolve_device(config.device)
    lineage = execution_lineage(scope, device, clean_tree_required=require_clean_tree)
    manifest = config.manifest
    root = lineage_root(checkpoint_base, lineage)
    paths = _artifact_paths(root, config.name, manifest.num_folds)
    occupied = sorted(str(p) for p in paths.values() if p.exists())
    if occupied:  # before any training: a frozen artifact is never silently replaced
        raise ArtifactExistsError(f"refusing to run: F1 artifacts already exist at {occupied}")

    logger = RunLogger(
        config.name,
        config.loggable(),
        root_dir=output_dir,
        device=device,
        metadata={
            **dict(metadata or {}),
            "execution_scope": scope,
            "lineage": lineage,
            "checkpoint_root": str(root),
        },
        require_clean_tree=require_clean_tree,
    )
    run_id = logger.run_dir.name
    if pool is None:
        pool = load_development_pool(config)

    def log_epoch(fit_name: str, horizon: int) -> Callable[[int, dict[str, Any]], None]:
        def write(epoch: int, record: dict[str, Any]) -> None:
            logger.log_metrics({"fit": fit_name, "horizon": horizon, **record}, step=epoch)

        return write

    horizon = INITIAL_HORIZON_EPOCHS
    decisions: list[HorizonDecision] = []
    history: list[dict[str, Any]] = []
    snapshot_root = Path(logger.run_dir) / "snapshots"
    while True:
        fits = [
            trainer(
                config,
                pool,
                train_subjects=manifest.training_subjects_for_fold(k),
                validation_subjects=manifest.validation_subjects(k),
                epochs=horizon,
                snapshot_dir=snapshot_root / f"horizon{horizon}" / f"fold{k}",
                device=device,
                on_epoch=log_epoch(f"fold{k}", horizon),
            )
            for k in range(manifest.num_folds)
        ]
        selections = [select_epoch(fit.raw_scores) for fit in fits]
        budget = common_budget([s.epoch for s in selections], expected_folds=manifest.num_folds)
        decision = decide_horizon(budget, horizon)
        decisions.append(decision)
        history.append(
            {
                "horizon": horizon,
                "selections": [s.to_dict() for s in selections],
                "decision": decision.to_dict(),
            }
        )
        if decision.action != "escalate":
            break
        # D19: the 30-epoch runs become diagnostics; all eight folds restart from scratch.
        if prune_snapshots:
            shutil.rmtree(snapshot_root / f"horizon{horizon}", ignore_errors=True)
        assert decision.next_horizon is not None
        horizon = decision.next_horizon

    if decision.action == "block":
        logger.finalize({"status": "blocked_pending_owner_decision", "horizons": history})
        raise HorizonBlockedError(
            f"the common budget equals the escalated {horizon}-epoch horizon again (D19). No refit "
            "or final-test prediction was made; this needs a new owner decision."
        )

    artifacts: dict[str, Path] = {}
    fold_records: list[dict[str, Any]] = []
    out_of_fold: list[Predictions] = []
    git = {k: logger.state[k] for k in ("git_commit", "git_dirty", "git_diff_sha256")}
    config_digests = {"run_config_sha256": logger.state["config_sha256"]}
    if "config_file_sha256" in logger.state["metadata"]:
        config_digests["config_file_sha256"] = logger.state["metadata"]["config_file_sha256"]
    for k, (fit, selection) in enumerate(zip(fits, selections, strict=True)):
        validation = manifest.validation_subjects(k)
        training = manifest.training_subjects_for_fold(k)
        validation_index = pool.index(validation)
        validation_recordings = pool.recordings_for(validation)
        common = {
            "source_run_id": run_id,
            "fold_index": k,
            "accepted_horizon": horizon,
            # The campaign's B on every role; best.pt's own epoch stays in source_epoch.
            "common_budget": budget,
            "manifest_hash": manifest.manifest_hash,
            "training_subject_ids": list(training),
            "scoring_subject_ids": list(validation),
            "normalization_sha256": config_digest(fit.normalization.to_dict()),
            "class_weights_sha256": class_weights_sha256(fit.class_weights),
            "class_weights_digest_encoding": CLASS_WEIGHTS_DIGEST_ENCODING,
            "source_window_identity_sha256": fit.window_identity_sha256,
            "config_digest": logger.state["config_sha256"],
            "training_rng_provenance": fit.rng_provenance,
            "device": str(device),
            "lineage": lineage,
            **git,
        }

        best = load_snapshot(config, fit, selection.epoch).to(device)
        best_predictions = predict(
            best,
            validation_index,
            validation_recordings,
            fit.normalization,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        # An analysis model's failed T blocks scaling for that model only, with no T=1 fallback.
        own_calibration: dict[str, Any]
        own_fit: TemperatureFit | None = None
        try:
            fitted = fit_population_temperature(
                best_predictions.logits,
                best_predictions.labels,
                best_predictions.subjects,
                best_predictions.repetitions,
            )
        except TemperatureFitError as error:
            own_calibration = {
                "method": "temperature_scaling",
                "status": "fit_failed",
                "reason": str(error),
            }
        else:
            own_fit = fitted
            own_calibration = _calibration_record(
                fitted,
                best_predictions,
                source="own_outer_fold_predictions_after_checkpoint_selection",
                role="selected_fold_extended_analysis_only",
                manifest_hash=manifest.manifest_hash,
                source_trajectory_provenance=[
                    {
                        "fold_index": k,
                        "source_run_id": run_id,
                        "source_epoch": selection.epoch,
                        "source_checkpoint_sha256": fit.snapshots[selection.epoch][1],
                        "accepted_horizon": horizon,
                    }
                ],
                config_digests=config_digests,
            )
        save_checkpoint(
            paths[f"fold{k}_best"],
            best.cpu(),
            normalization=fit.normalization.to_dict(),
            probability_calibration=own_calibration,
            provenance={
                **common,
                "role": SELECTED_FOLD_ANALYSIS,
                "source_epoch": selection.epoch,  # its individually selected epoch, not B
                "selection_smoothed_score": selection.smoothed_score,
                "source_checkpoint_sha256": fit.snapshots[selection.epoch][1],
            },
            exclusive=True,
        )

        fixed = load_snapshot(config, fit, budget).to(device)
        fixed_predictions = predict(
            fixed,
            validation_index,
            validation_recordings,
            fit.normalization,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        save_checkpoint(
            paths[f"fold{k}_fixed_budget"],
            fixed.cpu(),
            normalization=fit.normalization.to_dict(),
            provenance={
                **common,
                "role": COMMON_BUDGET_DEVELOPMENT,
                "source_epoch": budget,
                "training_budget": budget,
                "source_checkpoint_sha256": fit.snapshots[budget][1],
            },
            exclusive=True,
        )
        out_of_fold.append(fixed_predictions)
        for role in ("best", "fixed_budget"):
            artifacts[f"fold{k}_{role}"] = paths[f"fold{k}_{role}"]
            logger.add_artifact(f"fold{k}_{role}", paths[f"fold{k}_{role}"])
        best_identity = {
            "role": SELECTED_FOLD_ANALYSIS,
            "fold_index": k,
            "checkpoint_sha256": checkpoint_sha256(paths[f"fold{k}_best"]),
            "source_run_id": run_id,
            "source_epoch": selection.epoch,
            "source_checkpoint_sha256": fit.snapshots[selection.epoch][1],
            "common_budget": budget,
            "accepted_horizon": horizon,
        }
        fold_records.append(
            {
                "fold": k,
                "selected_epoch": selection.epoch,
                "selection_smoothed_score": selection.smoothed_score,
                "selected_checkpoint_sha256": best_identity["checkpoint_sha256"],
                "validation_metrics_selected_checkpoint": _validation_metrics(
                    best_predictions, config.num_classes
                ),
                "selected_fold_temperature": None if own_fit is None else own_fit.temperature,
                "cross_fitted_validation_diagnostics": _cross_fitted_diagnostics(
                    best_predictions, model_identity=best_identity
                ),
            }
        )

    pooled = _concatenate(out_of_fold)
    refit_fit = fit_population_temperature(
        pooled.logits, pooled.labels, pooled.subjects, pooled.repetitions
    )
    fold_digests = [checkpoint_sha256(paths[f"fold{k}_fixed_budget"]) for k in range(len(fits))]

    refit = trainer(
        config,
        pool,
        train_subjects=list(manifest.training_subjects),
        validation_subjects=(),
        epochs=budget,
        snapshot_dir=None,
        device=device,
        on_epoch=log_epoch(REFIT, budget),
    )
    save_checkpoint(
        paths["refit"],
        refit.model.cpu(),
        normalization=refit.normalization.to_dict(),
        probability_calibration=_calibration_record(
            refit_fit,
            pooled,
            source="pooled_subject_out_of_fold_fixed_budget_models",
            fold_artifact_sha256=fold_digests,
            role="refit_population_temperature",
            manifest_hash=manifest.manifest_hash,
            source_trajectory_provenance=[
                {
                    "fold_index": k,
                    "source_run_id": run_id,
                    "source_epoch": budget,
                    "source_checkpoint_sha256": fit.snapshots[budget][1],
                    "accepted_horizon": horizon,
                }
                for k, fit in enumerate(fits)
            ],
            config_digests=config_digests,
        ),
        provenance={
            "role": ALL_32_REFIT,
            "source_run_id": run_id,
            "training_budget": budget,
            "common_budget": budget,
            "accepted_horizon": horizon,
            "manifest_hash": manifest.manifest_hash,
            "training_subject_ids": list(manifest.training_subjects),
            "normalization_sha256": config_digest(refit.normalization.to_dict()),
            "class_weights_sha256": class_weights_sha256(refit.class_weights),
            "class_weights_digest_encoding": CLASS_WEIGHTS_DIGEST_ENCODING,
            "source_window_identity_sha256": refit.window_identity_sha256,
            "config_digest": logger.state["config_sha256"],
            "training_rng_provenance": refit.rng_provenance,
            "device": str(device),
            "lineage": lineage,
            **git,
        },
        exclusive=True,
    )
    artifacts["refit"] = paths["refit"]
    logger.add_artifact("refit", paths["refit"])

    if prune_snapshots:
        shutil.rmtree(snapshot_root, ignore_errors=True)
    summary = {
        "status": "complete",
        "lineage": lineage,
        "accepted_horizon": horizon,
        "common_budget": budget,
        "horizons": history,
        "folds": fold_records,
        "refit_temperature": refit_fit.to_dict(),
        "refit_epochs": refit.epochs,
        "final_test_evaluated": False,
    }
    logger.finalize(summary)
    return SubjectCVResult(
        run_dir=Path(logger.run_dir),
        lineage=lineage,
        accepted_horizon=horizon,
        budget=budget,
        selections=selections,
        decisions=decisions,
        artifacts=artifacts,
        refit_temperature=refit_fit,
        summary=summary,
    )

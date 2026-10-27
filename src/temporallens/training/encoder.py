"""The F0/F1 encoder training run: one configuration in, one recorded result out.

This is the path F0 exists to validate — loading, windowing, splitting, normalizing, the forward
pass, the loss, the metrics, the JSON log. F0 is explicitly **not a measurement**, so a debug run
is forbidden from writing ``refit.pt``: that filename is the artifact both arms load, and F0 must
not be able to produce it.

Three protocol rules are enforced here rather than left to the caller, because each fails
silently when it goes wrong:

* **The fitting set is the training subjects.** Normalization statistics and class weights are
  both fitted after the split, from training windows only. §3.3 calls this the most common silent
  leak in the project, and it leaves no trace in the logs.
* **No resampling.** Imbalance is handled by weighting the loss (D11). The result reports both
  the total and the unique training-window count so a test can assert they match.
* **No early stopping.** The epoch count is a declared constant (D18/D19); the loop runs it out.

``EncoderRunConfig`` takes explicit subject lists rather than a config path. That is deliberate:
there is no code path here that can resolve "every subject", so the leakage rule is testable
rather than merely documented. Translating a YAML config and a split manifest into these fields is
``scripts/train_encoder.py``'s job.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
    NUM_SUBJECTS,
    SubjectRecording,
    load_processed,
    processed_path,
)
from temporallens.data.windows import WindowIndex, concatenate, index_recording, segment_spans
from temporallens.evaluation.metrics import (
    accuracy,
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    mean_confidence_when_wrong,
    overconfidence_error,
    per_class_precision,
    per_class_recall,
    per_subject,
)
from temporallens.models.checkpoint import save_checkpoint
from temporallens.models.encoders.cnn1d import MODEL_TYPE, build_model
from temporallens.preprocessing.normalize import fit_channel_stats
from temporallens.training.class_weights import inverse_frequency_weights
from temporallens.utils.device import SUPPORTED_DEVICES, resolve_device
from temporallens.utils.run_logger import RunLogger, ensure_clean_tree

#: The only loss D11 permits for these runs.
SUPPORTED_LOSS = "class_weighted_cross_entropy"

#: The only normalization policy this trainer implements (D31): per-channel statistics fitted over
#: the samples training windows cover, each counted once, on the training *subjects*. F2's random
#: window split needs `train_split_covered_sample_stats` instead — a different fitting set, and
#: not written — so declaring it here fails rather than silently using the subject-based one.
SUPPORTED_NORMALIZATION = "train_subject_covered_sample_stats"


def _is_int(value: Any) -> bool:
    """True for a real integer. ``bool`` is an ``int`` subclass, and YAML's ``true`` is one."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_positive_int(name: str, value: Any) -> None:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_non_negative_int(name: str, value: Any) -> None:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _require_subjects(name: str, subjects: Any) -> None:
    """Subjects named once each, as DB2 subject numbers. Repeats matter because recordings are
    loaded into a dictionary keyed by subject, which would silently collapse them."""
    if not isinstance(subjects, (list, tuple)) or not all(
        _is_int(s) and 1 <= s <= NUM_SUBJECTS for s in subjects
    ):
        raise ValueError(f"{name} must be subject numbers in 1..{NUM_SUBJECTS}, got {subjects!r}")
    repeated = sorted({s for s in subjects if list(subjects).count(s) > 1})
    if repeated:
        raise ValueError(
            f"{name} repeats subject(s) {repeated}; each subject is one recording, so a repeat "
            "would be silently collapsed and the run would use fewer subjects than declared"
        )


@dataclass(frozen=True)
class EncoderRunConfig:
    """Everything one encoder run needs, with the subject roles stated explicitly."""

    name: str
    seed: int
    data_dir: Path
    output_dir: Path
    train_subjects: list[int]
    held_out_subjects: list[int]
    window_size: int
    stride: int
    input_channels: int
    num_classes: int
    hidden_dim: int
    embedding_dim: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    #: D11 permits exactly one loss here. It is a field rather than an assumption so that a
    #: config declaring something else fails loudly instead of being silently overridden.
    loss: str = SUPPORTED_LOSS
    #: Validated for the same reason as `loss`: these name what the run did, so a value this code
    #: cannot honour has to fail loudly instead of being quietly overruled.
    model_type: str = MODEL_TYPE
    normalize: str = SUPPORTED_NORMALIZATION
    device: str = "auto"
    max_windows_per_subject: int | None = None
    weight_decay: float = 0.0
    num_workers: int = 0
    save_checkpoint: bool = False
    save_predictions: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        _require_non_negative_int("seed", self.seed)
        _require_subjects("train_subjects", self.train_subjects)
        _require_subjects("held_out_subjects", self.held_out_subjects)
        for name in (
            "window_size",
            "stride",
            "hidden_dim",
            "embedding_dim",
            "batch_size",
            "epochs",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.stride > self.window_size:
            raise ValueError(
                f"stride ({self.stride}) must not exceed window_size ({self.window_size}); a larger "
                "stride skips the samples between windows, which no protocol here intends"
            )
        if self.input_channels != NUM_CHANNELS or not _is_int(self.input_channels):
            raise ValueError(
                f"input_channels must be {NUM_CHANNELS}: DB2 Exercise B has {NUM_CHANNELS} "
                f"channels, got {self.input_channels!r}"
            )
        if self.num_classes != NUM_CLASSES or not _is_int(self.num_classes):
            raise ValueError(
                f"num_classes must be {NUM_CLASSES}: DB2 Exercise B has {NUM_CLASSES} classes "
                f"(17 gestures and rest), got {self.num_classes!r}"
            )
        if not _is_finite_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be a positive finite number, got {self.learning_rate!r}"
            )
        if not _is_finite_number(self.weight_decay) or self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be a non-negative finite number, got {self.weight_decay!r}"
            )
        if not _is_finite_number(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be a finite number in [0, 1), got {self.dropout!r}")
        if self.max_windows_per_subject is not None:
            _require_positive_int("max_windows_per_subject", self.max_windows_per_subject)
        _require_non_negative_int("num_workers", self.num_workers)
        for name in ("save_checkpoint", "save_predictions"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false, got {getattr(self, name)!r}")
        overlap = set(self.train_subjects) & set(self.held_out_subjects)
        if overlap:
            raise ValueError(
                f"subject(s) {sorted(overlap)} are both training and held out; a split cannot "
                "overlap"
            )
        if not self.train_subjects or not self.held_out_subjects:
            raise ValueError("both train_subjects and held_out_subjects must be non-empty")
        if self.loss != SUPPORTED_LOSS:
            raise ValueError(
                f"loss must be {SUPPORTED_LOSS!r}, got {self.loss!r}. D11 handles imbalance in "
                "the loss and never by resampling, so an unweighted loss here would silently "
                "disagree with the config that declared it."
            )
        if self.model_type != MODEL_TYPE:
            raise ValueError(
                f"model_type must be {MODEL_TYPE!r}, got {self.model_type!r}; this trainer builds "
                "no other architecture, and training a CNN for a config that asked for something "
                "else would report a number for a model nobody ran"
            )
        if self.normalize != SUPPORTED_NORMALIZATION:
            raise ValueError(
                f"normalize must be {SUPPORTED_NORMALIZATION!r}, got {self.normalize!r} (D31)"
            )
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(
                f"device must be one of {list(SUPPORTED_DEVICES)}, got {self.device!r}"
            )


@dataclass
class EncoderRunResult:
    """What a run produced, including the facts a leakage test needs to check."""

    run_dir: Path
    metrics: dict[str, Any]
    epoch_losses: list[float]
    epochs_run: int
    train_subjects: tuple[int, ...]
    evaluation_subjects: tuple[int, ...]
    normalization: dict[str, Any]
    class_weights: list[float]
    train_window_count: int
    unique_train_window_count: int
    boundary_crossing_windows: int
    checkpoint_path: Path | None = None
    per_subject_metrics: dict[str, dict[int, float]] = field(default_factory=dict)


def _load(config: EncoderRunConfig, subjects: list[int]) -> dict[int, SubjectRecording]:
    return {s: load_processed(processed_path(config.data_dir, s)) for s in subjects}


def _index(config: EncoderRunConfig, recordings: dict[int, SubjectRecording]) -> WindowIndex:
    return concatenate(
        [
            index_recording(
                recording,
                window_size=config.window_size,
                stride=config.stride,
                max_windows=config.max_windows_per_subject,
                seed=config.seed,
            )
            for recording in recordings.values()
        ]
    )


def _count_boundary_crossing(index: WindowIndex, recordings: dict[int, SubjectRecording]) -> int:
    """Independent audit of D24: re-derive the segments rather than trust the index.

    Vectorised by binary search on segment starts. The obvious nested loop is
    O(windows x segments) -- about 3 s per 28-subject fold, and 95x slower for the same answer.
    """
    crossing = 0
    for subject, recording in recordings.items():
        spans = np.asarray(segment_spans(recording.label, recording.repetition), dtype=np.int64)
        starts = index.start[index.subject == subject]
        if starts.size == 0:
            continue
        # The only candidate is the last segment beginning at or before the window start.
        candidate = np.searchsorted(spans[:, 0], starts, side="right") - 1
        inside = (candidate >= 0) & (
            starts + index.window_size <= spans[np.clip(candidate, 0, len(spans) - 1), 1]
        )
        crossing += int((~inside).sum())
    return crossing


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    model.eval()
    probabilities, labels, subjects = [], [], []
    with torch.no_grad():
        for windows, label, subject in loader:
            logits = model(windows.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(label.numpy())
            subjects.append(subject.numpy())
    return (
        np.concatenate(probabilities).astype(np.float64),
        np.concatenate(labels).astype(np.int64),
        np.concatenate(subjects).astype(np.int64),
    )


def _weighted_epoch_loss(batches: list[tuple[float, float]]) -> float:
    """The epoch's class-weighted loss from ``(batch mean loss, batch weight sum)`` pairs.

    With class weights, PyTorch's batch mean divides by the batch's *weight* sum, not its size,
    so the objective over the whole epoch is each batch mean times its weight sum, summed and
    divided by the total weight. Averaging batch means equally instead over-counts rest-heavy
    batches, whose weight sum is small. Logged only: epochs are selected on validation macro-F1.
    """
    total_weight = sum(weight for _, weight in batches)
    if total_weight <= 0:
        raise ValueError("an epoch with no class weight cannot have a weighted mean loss")
    return sum(loss * weight for loss, weight in batches) / total_weight


def train_encoder(
    config: EncoderRunConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
    require_clean_tree: bool = False,
) -> EncoderRunResult:
    """Train the encoder and head on ``train_subjects``, score on ``held_out_subjects``.

    ``metadata`` is what the caller knows that the run config does not: the source config file,
    its mode and protocol state. ``require_clean_tree`` refuses to start from uncommitted code,
    which a reportable run must; see ``utils.run_logger``.
    """
    if require_clean_tree:  # before loading anything, so a refusal is immediate
        ensure_clean_tree(config.name)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)

    train_recordings = _load(config, config.train_subjects)
    eval_recordings = _load(config, config.held_out_subjects)
    train_index = _index(config, train_recordings)
    eval_index = _index(config, eval_recordings)

    # Fitted after the split, from training windows only (§3.3).
    normalization = fit_channel_stats(
        train_recordings,
        train_index,
        fitted_on=f"train_subjects={sorted(config.train_subjects)}",
    )
    class_weights = inverse_frequency_weights(train_index.label, num_classes=config.num_classes)

    train_loader = DataLoader(
        WindowDataset(train_index, train_recordings, normalization=normalization),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    eval_loader = DataLoader(
        WindowDataset(eval_index, eval_recordings, normalization=normalization),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(
        {
            "type": config.model_type,
            "input_channels": config.input_channels,
            "hidden_dim": config.hidden_dim,
            "embedding_dim": config.embedding_dim,
            "num_classes": config.num_classes,
            "dropout": config.dropout,
        }
    ).to(device)
    weight_tensor = torch.from_numpy(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # The config records what was requested (possibly `auto`); the log records what `auto` picked.
    logger = RunLogger(
        config.name,
        _loggable(config),
        root_dir=config.output_dir,
        device=device,
        metadata=metadata,
        require_clean_tree=require_clean_tree,
    )

    epoch_losses: list[float] = []
    for epoch in range(config.epochs):  # fixed horizon, no early stopping (D18/D19)
        model.train()
        batch_losses: list[tuple[float, float]] = []
        for windows, labels, _ in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(windows.to(device)), labels)
            loss.backward()
            optimizer.step()
            batch_losses.append((float(loss.detach()), float(weight_tensor[labels].sum())))
        mean_loss = _weighted_epoch_loss(batch_losses)
        epoch_losses.append(mean_loss)
        logger.log_metrics({"train_loss": mean_loss}, step=epoch)

    probabilities, labels, subjects = _evaluate(model, eval_loader, device)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels

    metrics: dict[str, Any] = {
        "accuracy": accuracy(predicted, labels),
        "macro_f1": macro_f1(predicted, labels, num_classes=config.num_classes),
        "per_class_precision": per_class_precision(
            predicted, labels, num_classes=config.num_classes
        ).tolist(),
        "per_class_recall": per_class_recall(
            predicted, labels, num_classes=config.num_classes
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            predicted, labels, num_classes=config.num_classes
        ).tolist(),
        "expected_calibration_error": expected_calibration_error(confidence, correct),
        "brier_score": brier_score(probabilities, labels),
        "overconfidence_error": overconfidence_error(confidence, correct),
        "mean_confidence_when_wrong": mean_confidence_when_wrong(confidence, correct),
    }
    per_subject_metrics = {
        "per_subject_accuracy": per_subject(accuracy, predicted, labels, subjects=subjects),
        "per_subject_macro_f1": per_subject(
            macro_f1, predicted, labels, subjects=subjects, num_classes=config.num_classes
        ),
    }

    checkpoint_path: Path | None = None
    if config.save_checkpoint:
        # F0 is not a measurement, so it may not write the name downstream arms load (D7).
        checkpoint_path = Path(logger.run_dir) / "encoder.pt"
        save_checkpoint(checkpoint_path, model, normalization=normalization.to_dict())
        logger.add_artifact("checkpoint", checkpoint_path)

    if config.save_predictions:
        predictions_path = Path(logger.run_dir) / "predictions.npz"
        np.savez_compressed(
            predictions_path,
            probs=probabilities.astype(np.float32),
            label=labels,
            subject=subjects,
        )
        logger.add_artifact("predictions", predictions_path)

    logger.finalize({"metrics": metrics, "per_subject": per_subject_metrics})

    unique_windows = len(
        set(zip(train_index.subject.tolist(), train_index.start.tolist(), strict=True))
    )
    return EncoderRunResult(
        run_dir=Path(logger.run_dir),
        metrics=metrics,
        epoch_losses=epoch_losses,
        epochs_run=config.epochs,
        train_subjects=tuple(sorted(config.train_subjects)),
        evaluation_subjects=tuple(sorted(config.held_out_subjects)),
        normalization=normalization.to_dict(),
        class_weights=[float(w) for w in class_weights],
        train_window_count=len(train_index),
        unique_train_window_count=unique_windows,
        boundary_crossing_windows=_count_boundary_crossing(train_index, train_recordings),
        checkpoint_path=checkpoint_path,
        per_subject_metrics=per_subject_metrics,
    )


def _loggable(config: EncoderRunConfig) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(config).items()
    }

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader

from temporallens.data.dataset import WindowDataset
from temporallens.data.ninapro import SubjectRecording, load_processed, processed_path
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
from temporallens.models.encoders.cnn1d import Cnn1dClassifier
from temporallens.preprocessing.normalize import fit_channel_stats
from temporallens.training.class_weights import inverse_frequency_weights
from temporallens.utils.device import get_device
from temporallens.utils.run_logger import RunLogger


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
    max_windows_per_subject: int | None = None
    weight_decay: float = 0.0
    num_workers: int = 0
    save_checkpoint: bool = False
    save_predictions: bool = False

    def __post_init__(self) -> None:
        overlap = set(self.train_subjects) & set(self.held_out_subjects)
        if overlap:
            raise ValueError(
                f"subject(s) {sorted(overlap)} are both training and held out; a split cannot "
                "overlap"
            )
        if not self.train_subjects or not self.held_out_subjects:
            raise ValueError("both train_subjects and held_out_subjects must be non-empty")


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
    """Independent audit of D24, rather than trusting the windower that produced the index."""
    crossing = 0
    for subject, recording in recordings.items():
        spans = segment_spans(recording.label, recording.repetition)
        starts = index.start[index.subject == subject]
        for start in starts:
            stop = int(start) + index.window_size
            inside = any(a <= start and stop <= b for a, b in spans)
            crossing += 0 if inside else 1
    return crossing


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
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


def train_encoder(config: EncoderRunConfig) -> EncoderRunResult:
    """Train the encoder and head on ``train_subjects``, score on ``held_out_subjects``."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = get_device()

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

    model = Cnn1dClassifier(
        input_channels=config.input_channels,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    logger = RunLogger(config.name, {"config": _loggable(config)}, root_dir=config.output_dir)

    epoch_losses: list[float] = []
    for epoch in range(config.epochs):  # fixed horizon, no early stopping (D18/D19)
        model.train()
        total, batches = 0.0, 0
        for windows, labels, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(windows.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        mean_loss = total / max(batches, 1)
        epoch_losses.append(mean_loss)
        logger.log_metrics({"train_loss": mean_loss}, step=epoch)

    probabilities, labels, subjects = _evaluate(model, eval_loader, device, config.num_classes)
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

    if config.save_predictions:
        np.savez_compressed(
            Path(logger.run_dir) / "predictions.npz",
            probs=probabilities.astype(np.float32),
            label=labels,
            subject=subjects,
        )

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

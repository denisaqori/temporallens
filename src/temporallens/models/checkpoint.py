"""The checkpoint contract: ``{model_state, model_config}``, rebuildable from the file alone.

AGENTS.md states this as a hard constraint and ``robustness_targets.yaml`` depends on it — a
target is just ``(name, artifact)`` with no ``model_type``, which only works because the config
travels inside the checkpoint. A single ``model_type`` string could not describe the adapter
stacks (encoder + projector + backbone + head) anyway.

Two keys, and only two. Extra top-level keys are how consumers start depending on things the
contract does not promise, so :func:`save_checkpoint` writes exactly those two and
:func:`load_checkpoint` refuses a file missing either.

Normalization statistics live inside ``model_config``. The encoder is frozen and reused by both
arms, so if its statistics did not travel with the file each arm would normalize differently and
the F1 reference row would stop being comparable — the same argument D23 makes for the
temperature. Saving without them is refused at write time rather than discovered downstream.

D7 distinguishes the two filenames: :func:`refit_checkpoint_path` is the artifact every consumer
reads, :func:`fold_checkpoint_path` is per-fold and for analysis only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from temporallens.models.encoders.cnn1d import MODEL_TYPE as CNN1D
from temporallens.models.encoders.cnn1d import build_model

CONTRACT_KEYS = frozenset({"model_state", "model_config"})

#: Rebuilders by ``model_config["type"]``. A type absent here is refused by name.
_BUILDERS = {CNN1D: build_model}


class CheckpointError(RuntimeError):
    """A checkpoint violates the contract, or names a model this process cannot rebuild."""


def refit_checkpoint_path(root: str | Path, experiment: str) -> Path:
    """``<root>/<experiment>/refit.pt`` — the one artifact downstream consumers read (D7)."""
    return Path(root) / experiment / "refit.pt"


def fold_checkpoint_path(root: str | Path, experiment: str, *, fold: int) -> Path:
    """``<root>/<experiment>/folds/fold<k>/best.pt`` — extended analysis only (D7).

    ``best`` keeps its ordinary meaning inside a fold: best epoch by validation macro-F1. The
    refit has no validation set, which is why it is not called ``best``.
    """
    return Path(root) / experiment / "folds" / f"fold{fold}" / "best.pt"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    normalization: Mapping[str, Any],
) -> None:
    """Write ``{model_state, model_config}``, with the statistics folded into the config."""
    if not normalization:
        raise CheckpointError(
            "refusing to save without normalization statistics: a consumer that rebuilds this "
            "model would normalize its inputs differently, so the checkpoint would be unusable "
            "downstream while appearing fine"
        )
    describe = getattr(model, "model_config", None)
    if not callable(describe):
        raise CheckpointError(
            f"{type(model).__name__} does not expose model_config(), so it cannot satisfy the "
            "checkpoint contract"
        )

    config = dict(describe())
    config["normalization"] = dict(normalization)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": config}, destination)


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(model_state, model_config)``, refusing anything off-contract."""
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError(f"{source}: expected a mapping, got {type(payload).__name__}")

    missing = sorted(CONTRACT_KEYS - payload.keys())
    if missing:
        raise CheckpointError(
            f"{source}: missing contract key(s) {missing}. Every checkpoint stores "
            "{model_state, model_config} so a consumer can rebuild from the file alone."
        )
    return payload["model_state"], payload["model_config"]


def rebuild_model(path: str | Path) -> nn.Module:
    """Reconstruct the model from the checkpoint alone — no external ``model_type``."""
    state, config = load_checkpoint(path)
    model_type = config.get("type")
    builder = _BUILDERS.get(model_type) if isinstance(model_type, str) else None
    if builder is None:
        raise CheckpointError(
            f"{path}: cannot rebuild model type {model_type!r}; known types are "
            f"{sorted(_BUILDERS)}"
        )
    model = builder(config)
    model.load_state_dict(state)
    return model

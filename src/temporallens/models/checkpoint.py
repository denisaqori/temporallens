"""The checkpoint contract: ``{model_state, model_config}``, rebuildable from the file alone.

AGENTS.md states this as a hard constraint and ``robustness_targets.yaml`` depends on it — a
target is just ``(name, artifact)`` with no ``model_type``, which only works because the config
travels inside the checkpoint. A single ``model_type`` string could not describe the adapter
stacks (encoder + projector + backbone + head) anyway.

Two keys, and only two. Extra top-level keys are how consumers start depending on things the
contract does not promise, so :func:`save_checkpoint` writes exactly those two and
:func:`load_checkpoint` refuses a file with any other set. Loading is as strict as the contract:
``model_state`` must be named tensors, ``model_config`` must carry the current
``schema_version`` and valid normalization statistics for the declared channel count, and the
file is read with ``weights_only=True`` so unpickling can never run code from it.

Normalization statistics live inside ``model_config``. The encoder is frozen and reused by both
arms, so if its statistics did not travel with the file each arm would normalize differently and
the F1 reference row would stop being comparable — the same argument D23 makes for the
temperature. Saving without them is refused at write time rather than discovered downstream.

D7 distinguishes the two filenames: :func:`refit_checkpoint_path` is the artifact every consumer
reads, :func:`fold_checkpoint_path` is per-fold and for analysis only.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from temporallens.models.encoders.cnn1d import MODEL_TYPE as CNN1D
from temporallens.models.encoders.cnn1d import rebuild_from_config
from temporallens.preprocessing.normalize import ChannelStats

CONTRACT_KEYS = frozenset({"model_state", "model_config"})

#: Bumped whenever ``model_config`` gains or changes a field a consumer relies on, so an old file
#: is refused by name rather than misread. Version 1: type, constructor arguments, recorded
#: architecture, normalization; ``probability_calibration`` is optional until the F1 harness fixes
#: its fields (D33).
MODEL_CONFIG_SCHEMA_VERSION = 1

#: Rebuilders by ``model_config["type"]``. A type absent here is refused by name. These are the
#: *checkpoint* rebuilders, which require the recorded architecture; fresh construction uses
#: ``build_model`` directly.
_BUILDERS = {CNN1D: rebuild_from_config}


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
    """Write ``{model_state, model_config}``, with the statistics folded into the config.

    The assembled config passes the loader's own validation before anything is written, so this
    cannot produce a file :func:`load_checkpoint` refuses — 11-channel statistics for a 12-channel
    model, say.
    """
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
    config["schema_version"] = MODEL_CONFIG_SCHEMA_VERSION

    destination = Path(path)
    _validate_model_config(config, f"refusing to save {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": config}, destination)


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(model_state, model_config)``, refusing anything off-contract."""
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise CheckpointError(
            f"{source}: contains objects other than tensors and plain data, so loading it would "
            f"have to run code from the file: {error}"
        ) from error
    except (OSError, RuntimeError, EOFError, ValueError) as error:
        raise CheckpointError(f"{source}: cannot read checkpoint: {error}") from error
    if not isinstance(payload, dict):
        raise CheckpointError(f"{source}: expected a mapping, got {type(payload).__name__}")

    missing = sorted(CONTRACT_KEYS - payload.keys())
    if missing:
        raise CheckpointError(
            f"{source}: missing contract key(s) {missing}. Every checkpoint stores "
            "{model_state, model_config} so a consumer can rebuild from the file alone."
        )
    extra = sorted(str(key) for key in payload.keys() - CONTRACT_KEYS)
    if extra:
        raise CheckpointError(
            f"{source}: unexpected top-level key(s) {extra}; the contract is exactly "
            "{model_state, model_config}"
        )

    state, config = payload["model_state"], payload["model_config"]
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise CheckpointError(f"{source}: model_state must map parameter names to tensors")
    if not isinstance(config, dict):
        raise CheckpointError(f"{source}: model_config must be a mapping")
    _validate_model_config(config, str(source))
    return dict(state), config


def _validate_model_config(config: dict[str, Any], source: str) -> None:
    # One validator for both ends of the contract; ``source`` prefixes its messages.
    version = config.get("schema_version")
    if isinstance(version, bool) or version != MODEL_CONFIG_SCHEMA_VERSION:
        raise CheckpointError(
            f"{source}: model_config schema_version is {version!r}, expected "
            f"{MODEL_CONFIG_SCHEMA_VERSION}; a file from another schema would be misread"
        )
    if not isinstance(config.get("type"), str):
        raise CheckpointError(f"{source}: model_config must name its model type")
    if "normalization" not in config:
        raise CheckpointError(
            f"{source}: model_config carries no normalization statistics, so a consumer would "
            "normalize its inputs differently from the run that trained it"
        )
    try:
        stats = ChannelStats.from_dict(config["normalization"])
    except (ValueError, TypeError) as error:
        raise CheckpointError(
            f"{source}: normalization statistics are malformed: {error}"
        ) from error
    channels = config.get("input_channels")
    if channels is not None and stats.mean.size != channels:
        raise CheckpointError(
            f"{source}: normalization statistics cover {stats.mean.size} channels but the model "
            f"takes {channels}"
        )
    calibration = config.get("probability_calibration")
    if calibration is not None and not isinstance(calibration, Mapping):
        raise CheckpointError(f"{source}: probability_calibration must be a mapping")


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
    try:
        model = builder(config)
    except (ValueError, KeyError) as error:
        raise CheckpointError(f"{path}: {error}") from error
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise CheckpointError(
            f"{path}: the stored weights do not match the model its config describes: {error}"
        ) from error
    return model

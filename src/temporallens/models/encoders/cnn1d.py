"""The Milestone-0 1-D CNN encoder and its classifier head.

The spec fixes the interface — ``cnn1d``, ``input_channels``, ``hidden_dim``, ``embedding_dim``,
``num_classes``, ``dropout`` — and nothing about the internals. Three Conv-BatchNorm-ReLU blocks
feed a global average pool and a linear projection to ``embedding_dim``; the pooling is what makes
the embedding size independent of the window length, so a debug config's different stride does not
change the contract downstream consumers rely on.

``freeze()`` implements AGENTS.md's definition exactly: ``requires_grad=False`` *and* ``eval()``
mode. The ``eval()`` half is not pedantry here — this encoder has BatchNorm, and a frozen encoder
left in train mode keeps updating its running statistics from whichever arm is using it, silently
changing a checkpoint both arms are supposed to share unchanged. ``train()`` is overridden so that
a later ``model.train()`` cannot quietly undo it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from torch import Tensor, nn

MODEL_TYPE = "cnn1d"

#: Frozen by D32. These are not per-run knobs — they are what the name ``cnn1d`` *means*. If they
#: were config keys, two runs could both declare ``model.type: cnn1d`` and be different
#: architectures, which would make the name worthless and break the checkpoint contract's promise
#: that ``model_config`` identifies the architecture. Changing any of these is a new model type
#: with a new name, not a new value here.
ARCHITECTURE: Mapping[str, Any] = MappingProxyType(
    {
        "block_kernels": (7, 5, 3),
        "pool_factors": (4, 4),
        "width_multiplier": 2,
        "normalization": "batchnorm1d",
        "activation": "relu",
        "global_pool": "adaptive_avg",
    }
)


def _architecture_payload() -> dict[str, Any]:
    """``ARCHITECTURE`` as JSON-safe values, for recording into ``model_config``."""
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in ARCHITECTURE.items()
    }


def _normalised(architecture: Mapping[str, Any]) -> dict[str, Any]:
    """Compare recorded against frozen without tripping over tuple-versus-list."""
    return {
        key: list(value) if isinstance(value, (list, tuple)) else value
        for key, value in architecture.items()
    }


def _block(in_channels: int, out_channels: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class Cnn1dEncoder(nn.Module):
    """Maps a raw EMG window ``(batch, channels, length)`` to an ``embedding_dim`` vector."""

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout
        self._frozen = False

        first, second, third = ARCHITECTURE["block_kernels"]
        pool_one, pool_two = ARCHITECTURE["pool_factors"]
        wide = hidden_dim * ARCHITECTURE["width_multiplier"]
        self.features = nn.Sequential(
            _block(input_channels, hidden_dim, first),
            nn.MaxPool1d(pool_one),
            _block(hidden_dim, wide, second),
            nn.MaxPool1d(pool_two),
            _block(wide, wide, third),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(wide, embedding_dim)

    def forward(self, windows: Tensor) -> Tensor:
        pooled = self.features(windows).flatten(1)
        return self.project(self.dropout(pooled))

    def freeze(self) -> None:
        """``requires_grad=False`` and ``eval()`` mode, and keep it that way."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._frozen = True
        self.eval()

    @property
    def frozen(self) -> bool:
        return self._frozen

    def train(self, mode: bool = True) -> Cnn1dEncoder:
        """A frozen encoder stays in eval mode even when its parent is put into train mode."""
        if self._frozen:
            return super().train(False)
        return super().train(mode)


class Cnn1dClassifier(nn.Module):
    """Encoder plus the classifier head trained with it in F0/F1.

    The encoder is exposed because both arms load this checkpoint and reuse it frozen; the head
    is theirs to replace.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Cnn1dEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_classes))

    def forward(self, windows: Tensor) -> Tensor:
        return self.head(self.encoder(windows))

    def model_config(self) -> dict[str, Any]:
        """Exactly the keyword arguments that reconstruct this model, plus its type.

        The checkpoint contract promises a consumer can rebuild from the file alone, so this has
        to stay in step with ``__init__``.
        """
        return {
            "type": MODEL_TYPE,
            "input_channels": self.encoder.input_channels,
            "hidden_dim": self.encoder.hidden_dim,
            "embedding_dim": self.encoder.embedding_dim,
            "num_classes": self.num_classes,
            "dropout": self.encoder.dropout_p,
            # Recorded, not configurable. A consumer can therefore tell whether the checkpoint it
            # holds was built by the same `cnn1d` this code defines — see build_model.
            "architecture": _architecture_payload(),
        }


def build_model(config: dict[str, Any]) -> Cnn1dClassifier:
    """Construct from a ``model_config`` payload, validating what it claims to be.

    Two checks, both guarding the checkpoint contract. A declared ``type`` other than ``cnn1d``
    means this builder is the wrong one, and silently building a CNN anyway is how a config gets
    to ask for one architecture and report another. A recorded ``architecture`` that disagrees
    with :data:`ARCHITECTURE` means the file was written by a different definition of this name;
    its weights would load into the wrong shapes, or worse, into right-shaped wrong layers.
    """
    declared = config.get("type")
    if declared is not None and declared != MODEL_TYPE:
        raise ValueError(
            f"build_model is the {MODEL_TYPE!r} builder but model_config declares {declared!r}"
        )
    recorded = config.get("architecture")
    if recorded is not None and _normalised(recorded) != _normalised(ARCHITECTURE):
        raise ValueError(
            f"model_config records a different {MODEL_TYPE!r} architecture than this code "
            f"defines: recorded {_normalised(recorded)}, expected {_normalised(ARCHITECTURE)}. "
            "A changed internal is a new model type with a new name (D32)."
        )
    kwargs = {
        key: config[key]
        for key in ("input_channels", "hidden_dim", "embedding_dim", "num_classes", "dropout")
    }
    return Cnn1dClassifier(**kwargs)

"""`EncoderRunConfig` refuses values that would train something other than what was declared.

Each case here was accepted before: zero epochs, a zero learning rate and a repeated subject all
produced a run, and the repeated subject was then silently deduplicated when recordings were loaded
into a dictionary — so the run trained on fewer subjects than its config listed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from temporallens.training.encoder import EncoderRunConfig


def _valid(**overrides: Any) -> EncoderRunConfig:
    base: dict[str, Any] = dict(
        name="debug_tiny",
        seed=42,
        data_dir=Path("data/processed/ninapro_db2"),
        output_dir=Path("results/runs"),
        train_subjects=[1, 2],
        held_out_subjects=[3],
        window_size=400,
        stride=200,
        input_channels=12,
        num_classes=18,
        hidden_dim=32,
        embedding_dim=64,
        dropout=0.1,
        batch_size=8,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_windows_per_subject=50,
    )
    base.update(overrides)
    return EncoderRunConfig(**base)


def test_the_real_f0_settings_are_valid() -> None:
    _valid()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("epochs", 0, "positive integer"),
        ("epochs", 2.5, "positive integer"),
        ("epochs", True, "positive integer"),  # YAML `true` is an int subclass in Python
        ("batch_size", 0, "positive integer"),
        ("window_size", 0, "positive integer"),
        ("hidden_dim", 0, "positive integer"),
        ("embedding_dim", -4, "positive integer"),
        ("stride", 0, "positive integer"),
        ("stride", 401, "stride"),
        ("input_channels", 8, "12 channels"),
        ("num_classes", 17, "18 classes"),
        ("learning_rate", 0.0, "positive finite"),
        ("learning_rate", -1e-3, "positive finite"),
        ("learning_rate", math.nan, "positive finite"),
        ("learning_rate", math.inf, "positive finite"),
        ("learning_rate", "3e-4", "positive finite"),  # PyYAML reads 3e-4 (no dot) as a string
        ("weight_decay", -1e-4, "non-negative finite"),
        ("weight_decay", math.nan, "non-negative finite"),
        ("dropout", -0.1, "dropout"),
        ("dropout", 1.0, "dropout"),
        ("dropout", math.nan, "dropout"),
        ("max_windows_per_subject", 0, "positive integer"),
        ("num_workers", -1, "non-negative integer"),
        ("seed", -1, "non-negative integer"),
        ("seed", 2**32, "below 2"),
        ("name", "", "name"),
        ("train_subjects", [1, 1, 2], "repeat"),
        ("held_out_subjects", [3, 3], "repeat"),
        ("train_subjects", [0, 2], "1..40"),
        ("held_out_subjects", [41], "1..40"),
        ("train_subjects", [1, True], "1..40"),
        ("save_checkpoint", "yes", "true or false"),
    ],
)
def test_an_invalid_value_is_refused(field: str, value: Any, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        _valid(**{field: value})

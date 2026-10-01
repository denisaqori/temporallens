from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from temporallens.utils.device import get_device
from temporallens.utils.run_logger import RunLogger, get_git_commit

# Anchor repo paths to this file so tests pass from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_device_selection_returns_available_backend() -> None:
    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}
    if device.type == "cuda":
        assert torch.cuda.is_available()
    if device.type == "mps":
        assert torch.backends.mps.is_available()


def test_debug_config_has_expected_local_shape() -> None:
    config_path = REPO_ROOT / "configs" / "experiment" / "foundation" / "debug_tiny.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["dataset"]["exercise"] == "B"
    assert config["dataset"]["input_channels"] == 12
    assert config["dataset"]["num_classes"] == 18
    assert config["training"]["device"] == "auto"


def test_run_logger_writes_local_json(tmp_path: Path) -> None:
    logger = RunLogger("smoke", {"seed": 42}, root_dir=tmp_path)
    logger.log_metrics(
        {
            "loss": np.float32(1.0),
            "confusion_matrix": np.asarray([[2, 1], [0, 3]], dtype=np.int64),
            "per_class_recall": torch.tensor([0.5, 1.0]),
        },
        step=0,
    )
    logger.finalize({"status": "ok", "artifact_root": Path("checkpoints")})

    state = json.loads(logger.log_path.read_text())
    assert state["summary"] == {"status": "ok", "artifact_root": "checkpoints"}
    metrics = json.loads(logger.metrics_path.read_text())
    assert metrics["metrics"]["loss"] == 1.0
    assert metrics["metrics"]["confusion_matrix"] == [[2, 1], [0, 3]]
    assert metrics["metrics"]["per_class_recall"] == [0.5, 1.0]


def test_git_commit_is_anchored_to_the_package_worktree(tmp_path: Path, monkeypatch) -> None:
    expected = get_git_commit(REPO_ROOT)
    monkeypatch.chdir(tmp_path)
    assert get_git_commit() == expected

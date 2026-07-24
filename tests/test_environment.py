from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from temporallens.utils.device import get_device
from temporallens.utils.run_logger import RunLogger

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
    logger.log_metrics({"loss": 1.0}, step=0)
    logger.finalize({"status": "ok"})

    state = json.loads(logger.log_path.read_text())
    assert state["summary"] == {"status": "ok"}
    assert logger.metrics_path.read_text().strip()

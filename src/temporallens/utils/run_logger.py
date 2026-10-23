"""Always-on local JSON logging for reproducible experiments."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_default(value: Any) -> Any:
    """Convert metric/config values that have an unambiguous JSON representation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, indent=indent, default=_json_default)


def get_git_commit(repo_dir: str | Path | None = None) -> str:
    """Return the current Git commit, or ``unknown`` before the first commit."""
    # In an editable install this module lives inside the active worktree, so anchoring the query
    # here avoids recording whichever unrelated repository happens to be the process cwd.
    search_from = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(search_from), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class RunLogger:
    """Record config, metrics, artifacts, and summaries without a network service.

    ``device`` is the device the run actually executes on, recorded at the top level beside the
    git commit because it describes the run, not its configuration. It is required so no run can be
    logged without it, and it must be resolved: ``"auto"`` is a request, and a log that says only
    ``auto`` cannot tell a reader whether the numbers came from CUDA, MPS or the CPU — which matters,
    because seeding does not make those backends numerically identical.
    """

    def __init__(
        self,
        experiment_name: str,
        config: dict[str, Any],
        root_dir: str | Path = "results/runs",
        *,
        device: torch.device | str,
    ) -> None:
        if str(device) == "auto":
            raise ValueError(
                "RunLogger needs the resolved device, not the 'auto' request; resolve it first "
                "with temporallens.utils.device.resolve_device"
            )
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir = Path(root_dir) / f"{timestamp}_{experiment_name}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.log_path = self.run_dir / "run.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.state: dict[str, Any] = {
            "run_id": self.run_dir.name,
            "experiment_name": experiment_name,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
            "device": str(device),
            "config": config,
            "summary": {},
            "artifacts": {},
        }
        self._write()

    def _write(self) -> None:
        temporary_path = self.log_path.with_suffix(".json.tmp")
        temporary_path.write_text(_json_dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.log_path)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record = {
            "time": datetime.now(UTC).isoformat(),
            "step": step,
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(_json_dumps(record) + "\n")

    def add_artifact(self, name: str, path: str | Path) -> None:
        self.state["artifacts"][name] = str(path)
        self._write()

    def finalize(self, summary: dict[str, Any]) -> None:
        self.state["summary"] = summary
        self.state["finished_at"] = datetime.now(UTC).isoformat()
        self._write()

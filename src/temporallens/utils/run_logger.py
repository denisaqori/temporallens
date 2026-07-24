"""Always-on local JSON logging for reproducible experiments."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_git_commit() -> str:
    """Return the current Git commit, or ``unknown`` before the first commit."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class RunLogger:
    """Record config, metrics, artifacts, and summaries without a network service."""

    def __init__(
        self,
        experiment_name: str,
        config: dict[str, Any],
        root_dir: str | Path = "results/runs",
    ) -> None:
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
            "config": config,
            "summary": {},
            "artifacts": {},
        }
        self._write()

    def _write(self) -> None:
        temporary_path = self.log_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.log_path)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record = {
            "time": datetime.now(UTC).isoformat(),
            "step": step,
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def add_artifact(self, name: str, path: str | Path) -> None:
        self.state["artifacts"][name] = str(path)
        self._write()

    def finalize(self, summary: dict[str, Any]) -> None:
        self.state["summary"] = summary
        self.state["finished_at"] = datetime.now(UTC).isoformat()
        self._write()

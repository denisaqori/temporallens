"""Always-on local JSON logging for reproducible experiments.

A run directory has to let someone trace its numbers to the code and config that produced them. A
commit hash alone cannot: the first F0 run recorded a commit from before ``train_encoder.py``
existed, because the script was still uncommitted. So the log also records whether the working
tree was dirty and a digest of the uncommitted changes, a digest of the exact config, the caller's
metadata, and a hash of every artifact the run wrote. A run that may be reported can refuse a
dirty tree outright.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
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


class UncommittedCodeError(RuntimeError):
    """A run that may be reported was started from a working tree that is not clean."""


def _repository_root(repo_dir: str | Path | None) -> Path | None:
    search_from = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    try:
        top = subprocess.check_output(
            ["git", "-C", str(search_from), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(top)


def get_git_state(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """The commit, whether the tree is dirty, what changed, and a digest of the changes.

    The digest covers the tracked diff against ``HEAD`` and the content of every untracked,
    non-ignored file, so two runs with the same commit and digest ran the same code. Like
    :func:`get_git_commit` it is anchored to this package's worktree, not the process's cwd.
    ``dirty`` is ``None`` when the state cannot be read.
    """
    unknown: dict[str, Any] = {
        "commit": "unknown",
        "dirty": None,
        "changes": [],
        "diff_sha256": None,
    }
    root = _repository_root(repo_dir)
    if root is None:
        return unknown
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).splitlines()
        diff = subprocess.check_output(
            ["git", "-C", str(root), "diff", "HEAD", "--binary"], stderr=subprocess.DEVNULL
        )
        untracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).split("\0")
    except (OSError, subprocess.CalledProcessError):
        return unknown
    changes = [line for line in changes if line]
    if not changes:
        return {"commit": commit, "dirty": False, "changes": [], "diff_sha256": None}
    digest = hashlib.sha256(diff)
    for name in sorted(path for path in untracked if path):
        digest.update(b"\0untracked\0" + name.encode())
        digest.update((root / name).read_bytes())
    return {"commit": commit, "dirty": True, "changes": changes, "diff_sha256": digest.hexdigest()}


def config_digest(config: Mapping[str, Any]) -> str:
    """SHA-256 of the config as canonical JSON, so key order does not change its identity."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_clean_tree(experiment_name: str, git: Mapping[str, Any] | None = None) -> None:
    """Refuse to proceed unless the working tree is known to be clean.

    Called at the start of a run so it fails before loading any data, and again by
    :class:`RunLogger` so no logger can be created for a reportable run from uncommitted code.
    """
    state = get_git_state() if git is None else git
    if state["dirty"] is False:
        return
    reason = (
        "uncommitted changes: " + "; ".join(state["changes"][:5])
        if state["dirty"]
        else "cannot confirm the working tree is clean"
    )
    raise UncommittedCodeError(
        f"refusing a reportable run of {experiment_name!r}: {reason}. Commit first, so the "
        "recorded commit identifies the code that produced the numbers."
    )


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
        metadata: Mapping[str, Any] | None = None,
        require_clean_tree: bool = False,
    ) -> None:
        if str(device) == "auto":
            raise ValueError(
                "RunLogger needs the resolved device, not the 'auto' request; resolve it first "
                "with temporallens.utils.device.resolve_device"
            )
        git = get_git_state()
        if require_clean_tree:
            ensure_clean_tree(experiment_name, git)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir = Path(root_dir) / f"{timestamp}_{experiment_name}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.log_path = self.run_dir / "run.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.state: dict[str, Any] = {
            "run_id": self.run_dir.name,
            "experiment_name": experiment_name,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "git_changes": git["changes"],
            "git_diff_sha256": git["diff_sha256"],
            "device": str(device),
            "config_sha256": config_digest(config),
            "config": config,
            "metadata": dict(metadata or {}),
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
        """Register a file the run wrote, with its hash and size, so it can be matched later."""
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(f"cannot register artifact {name!r}: no file at {artifact}")
        self.state["artifacts"][name] = {
            "path": str(artifact),
            "sha256": _file_sha256(artifact),
            "bytes": artifact.stat().st_size,
        }
        self._write()

    def finalize(self, summary: dict[str, Any]) -> None:
        self.state["summary"] = summary
        self.state["finished_at"] = datetime.now(UTC).isoformat()
        self._write()

"""Run provenance: what a run directory must record so its numbers can be traced to their code.

A commit hash alone cannot identify the code that ran. The F0 debug run recorded a commit from
before `train_encoder.py` existed, because the script was still uncommitted. So the log also
records whether the tree was dirty and a digest of the uncommitted changes, a digest of the exact
config it ran, the loader's metadata, and a hash of every artifact it wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from temporallens.utils import run_logger
from temporallens.utils.run_logger import RunLogger, get_git_commit, get_git_state

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN = {"commit": "a" * 40, "dirty": False, "changes": [], "diff_sha256": None}
DIRTY = {"commit": "a" * 40, "dirty": True, "changes": [" M src/x.py"], "diff_sha256": "f" * 64}


def _logger(tmp_path, monkeypatch, *, git=CLEAN, config=None, **kwargs) -> RunLogger:
    monkeypatch.setattr(run_logger, "get_git_state", lambda: dict(git))
    return RunLogger("smoke", config or {"seed": 42}, root_dir=tmp_path, device="cpu", **kwargs)


def _state(logger: RunLogger) -> dict:
    return json.loads(logger.log_path.read_text())


def test_the_working_tree_state_is_recorded_beside_the_commit(tmp_path, monkeypatch) -> None:
    state = _state(_logger(tmp_path, monkeypatch, git=DIRTY))
    assert state["git_commit"] == "a" * 40
    assert state["git_dirty"] is True
    assert state["git_changes"] == [" M src/x.py"]
    assert state["git_diff_sha256"] == "f" * 64


def test_a_reportable_run_from_a_dirty_tree_is_refused(tmp_path, monkeypatch) -> None:
    """A reportable number must be traceable to committed code, or nobody can rerun it."""
    with pytest.raises(run_logger.UncommittedCodeError, match="uncommitted changes"):
        _logger(tmp_path, monkeypatch, git=DIRTY, require_clean_tree=True)
    assert not any(tmp_path.iterdir()), "a refused run must not leave a run directory behind"


def test_an_unknown_tree_state_is_refused_for_a_reportable_run(tmp_path, monkeypatch) -> None:
    unknown = {"commit": "unknown", "dirty": None, "changes": [], "diff_sha256": None}
    with pytest.raises(run_logger.UncommittedCodeError, match="cannot confirm"):
        _logger(tmp_path, monkeypatch, git=unknown, require_clean_tree=True)


def test_a_debug_run_from_a_dirty_tree_is_allowed_and_records_it(tmp_path, monkeypatch) -> None:
    state = _state(_logger(tmp_path, monkeypatch, git=DIRTY, require_clean_tree=False))
    assert state["git_dirty"] is True


def test_a_reportable_run_from_a_clean_tree_is_allowed(tmp_path, monkeypatch) -> None:
    state = _state(_logger(tmp_path, monkeypatch, git=CLEAN, require_clean_tree=True))
    assert state["git_dirty"] is False and state["git_diff_sha256"] is None


def test_the_config_digest_identifies_the_logged_config(tmp_path, monkeypatch) -> None:
    """Key order is not identity: the digest is over canonical JSON."""
    first = _state(_logger(tmp_path / "a", monkeypatch, config={"seed": 42, "epochs": 2}))
    reordered = _state(_logger(tmp_path / "b", monkeypatch, config={"epochs": 2, "seed": 42}))
    changed = _state(_logger(tmp_path / "c", monkeypatch, config={"seed": 42, "epochs": 3}))
    canonical = json.dumps({"epochs": 2, "seed": 42}, sort_keys=True, separators=(",", ":"))
    assert first["config_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert first["config_sha256"] == reordered["config_sha256"]
    assert first["config_sha256"] != changed["config_sha256"]


def test_metadata_is_recorded(tmp_path, monkeypatch) -> None:
    metadata = {"experiment.mode": "debug", "declared_metrics": ["accuracy"]}
    assert _state(_logger(tmp_path, monkeypatch, metadata=metadata))["metadata"] == metadata


def test_an_artifact_is_registered_with_its_hash_and_size(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path / "runs", monkeypatch)
    artifact = tmp_path / "predictions.npz"
    artifact.write_bytes(b"not really an archive")
    logger.add_artifact("predictions", artifact)
    entry = _state(logger)["artifacts"]["predictions"]
    assert entry == {
        "path": str(artifact),
        "sha256": hashlib.sha256(b"not really an archive").hexdigest(),
        "bytes": len(b"not really an archive"),
    }


def test_a_missing_artifact_cannot_be_registered(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        logger.add_artifact("checkpoint", tmp_path / "missing.pt")


def test_the_real_tree_state_describes_this_repository() -> None:
    state = get_git_state()
    assert state["commit"] == get_git_commit(REPO_ROOT)
    assert isinstance(state["dirty"], bool)
    if state["dirty"]:
        assert state["changes"] and re.fullmatch(r"[0-9a-f]{64}", state["diff_sha256"])
    else:
        assert state["changes"] == [] and state["diff_sha256"] is None


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def committed_repo(tmp_path, monkeypatch) -> Path:
    """A one-commit repository, isolated from the user's global and system Git configuration."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("data/\n")
    (tmp_path / "tracked.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".gitignore", "tracked.py")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
    return tmp_path


@pytest.mark.parametrize("setting", ["no", "normal", "all"])
def test_untracked_code_is_dirty_whatever_git_is_configured_to_show(
    committed_repo, setting
) -> None:
    """`status.showUntrackedFiles=no` hides new files from a plain `git status`, and an
    uncommitted new module would then pass the clean-tree guard."""
    _git(committed_repo, "config", "status.showUntrackedFiles", setting)
    (committed_repo / "src").mkdir()
    (committed_repo / "src" / "new_module.py").write_text("y = 2\n")
    state = get_git_state(committed_repo)
    assert state["dirty"] is True
    assert "?? src/new_module.py" in state["changes"]
    with pytest.raises(run_logger.UncommittedCodeError, match="new_module.py"):
        run_logger.ensure_clean_tree("smoke", state)


def test_ignored_files_leave_the_tree_clean(committed_repo) -> None:
    """Data and run outputs are gitignored; they are not code, and must not block a run."""
    (committed_repo / "data").mkdir()
    (committed_repo / "data" / "subject_01.npz").write_bytes(b"not code")
    state = get_git_state(committed_repo)
    assert state["dirty"] is False and state["changes"] == []

"""Fail-closed plan resolution for the robustness-evaluation stub."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from temporallens.evaluation.robustness import (
    duplicate_target_names,
    invalid_shared_k0_aliases,
    pending_field_paths,
    source_protocol_blockers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_evaluate_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("temporallens_evaluate_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pending_field_paths_finds_nested_sentinels() -> None:
    target = {
        "name": "g3",
        "run_selection": "pending_immutable_run_id",
        "nested": {"aggregation": "pending_decision", "ready": "fixed"},
        "axes": ["subject", {"draw": "pending_seed_contract"}],
    }
    assert pending_field_paths(target) == [
        "run_selection",
        "nested.aggregation",
        "axes[1].draw",
    ]


def test_duplicate_target_names_are_rejected_independently_of_strategy() -> None:
    targets = [
        {"name": "same", "arm": "foundation"},
        {"name": "same", "arm": "generation"},
        {"name": "unique", "arm": "language"},
    ]
    assert duplicate_target_names(targets) == ["same"]


def test_shared_k0_aliases_must_resolve_to_registry_targets() -> None:
    targets = [
        {"name": "f1_encoder"},
        {"name": "real_adaptation", "shared_k0_with": "f1_encoder"},
        {"name": "broken", "shared_k0_with": "not_present"},
    ]
    assert invalid_shared_k0_aliases(targets) == {"broken": "not_present"}


def test_source_protocol_blockers_include_status_and_declared_paths(tmp_path: Path) -> None:
    source_config = tmp_path / "blocked.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "protocol": {
                    "status": "blocked_pending_decisions",
                    "blocked_on": ["evaluation.aggregation", "personalization.rest_policy"],
                }
            }
        )
    )
    assert source_protocol_blockers({"source_config": str(source_config)}) == [
        "source_config.protocol.status=blocked_pending_decisions",
        "source_config.evaluation.aggregation",
        "source_config.personalization.rest_policy",
    ]


def test_pending_target_never_dispatches_even_when_its_artifact_exists(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    evaluate = _load_evaluate_script()
    perturbation_path = tmp_path / "perturbation.yaml"
    perturbation_path.write_text(
        yaml.safe_dump({"perturbation": {"type": "noise", "levels": [0.1]}})
    )

    existing_run = tmp_path / "existing-run"
    existing_run.mkdir()
    registry_path = tmp_path / "targets.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "targets": [
                    {
                        "name": "blocked_g3",
                        "arm": "generation",
                        "run_dir": str(existing_run),
                        "run_selection": "pending_immutable_run_id",
                    }
                ]
            }
        )
    )

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("a target with a pending sentinel reached run_target")

    monkeypatch.setattr(evaluate, "run_target", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(perturbation_path),
            "--targets",
            str(registry_path),
            "--target",
            "blocked_g3",
        ],
    )

    assert evaluate.main() == 2
    output = capsys.readouterr().out
    assert "block   blocked_g3" in output
    assert "run_selection" in output
    assert "run     blocked_g3" not in output


def test_nonready_source_protocol_never_dispatches_existing_artifact(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    evaluate = _load_evaluate_script()
    perturbation_path = tmp_path / "perturbation.yaml"
    perturbation_path.write_text(
        yaml.safe_dump({"perturbation": {"type": "noise", "levels": [0.1]}})
    )
    source_config = tmp_path / "g3.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "protocol": {
                    "status": "blocked_pending_decisions",
                    "blocked_on": ["personalization.adaptation.objective.candidate_grid"],
                }
            }
        )
    )
    existing_run = tmp_path / "existing-run"
    existing_run.mkdir()
    registry_path = tmp_path / "targets.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "targets": [
                    {
                        "name": "blocked_g3",
                        "arm": "generation",
                        "source_config": str(source_config),
                        "run_dir": str(existing_run),
                    }
                ]
            }
        )
    )

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("a target with a non-ready source protocol reached run_target")

    monkeypatch.setattr(evaluate, "run_target", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(perturbation_path),
            "--targets",
            str(registry_path),
            "--target",
            "blocked_g3",
        ],
    )

    assert evaluate.main() == 2
    output = capsys.readouterr().out
    assert "source_config.protocol.status=blocked_pending_decisions" in output
    assert "source_config.personalization.adaptation.objective.candidate_grid" in output
    assert "run     blocked_g3" not in output


def test_unready_perturbation_refuses_before_any_target_is_considered(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An undefined perturbation is one refusal, not a per-target skip.

    A type and a level list are not an operational definition. If the application order,
    sampling, seeds, repetitions or aggregation are unresolved, every target would be scored
    against an undefined transform — so the axis itself is refused, and a ready target with a
    present artifact must not be reached.
    """
    evaluate = _load_evaluate_script()
    perturbation_path = tmp_path / "perturbation.yaml"
    perturbation_path.write_text(
        yaml.safe_dump(
            {
                "protocol": {
                    "status": "blocked_pending_decisions",
                    "blocked_on": ["perturbation.seed_derivation"],
                },
                "perturbation": {"type": "noise", "levels": [0.1], "seed_derivation": None},
            }
        )
    )
    checkpoint = tmp_path / "refit.pt"
    checkpoint.write_bytes(b"")
    registry_path = tmp_path / "targets.yaml"
    registry_path.write_text(
        yaml.safe_dump({"targets": [{"name": "ready_target", "checkpoint": str(checkpoint)}]})
    )

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("an unresolved perturbation reached run_target")

    monkeypatch.setattr(evaluate, "run_target", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--config", str(perturbation_path), "--targets", str(registry_path)],
    )

    with pytest.raises(SystemExit) as exit_info:
        evaluate.main()
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "is not protocol-ready" in message
    assert "perturbation.seed_derivation" in message
    assert "ready_target" not in message


def test_every_registry_source_config_declares_a_protocol_block() -> None:
    """Gating must not depend on absence.

    `protocol_blockers` returns nothing when a config has no `protocol:` block, so a target whose
    source config simply never declares one resolves as dispatchable. That is indistinguishable
    from a deliberate `status: ready` and is how four targets were silently dispatchable. Requiring
    the declaration makes readiness a statement someone made rather than a gap nobody noticed.
    """
    registry = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiment" / "robustness_targets.yaml").read_text()
    )
    missing = []
    for target in registry["targets"]:
        source = target.get("source_config")
        assert source, f"target {target['name']!r} declares no source_config"
        config = yaml.safe_load((REPO_ROOT / source).read_text())
        protocol = config.get("protocol")
        if not isinstance(protocol, dict) or "status" not in protocol:
            missing.append(f"{target['name']} -> {source}")
    assert not missing, f"registry source configs with no protocol.status: {missing}"

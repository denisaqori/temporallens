#!/usr/bin/env python3
"""Robustness evaluation driver (contract stub).

A robustness evaluation is a matrix: one perturbation (F3/F4/F5) applied to the input window,
evaluated against one or more trained TARGETS (the F1 encoder, the language-arm adapter stacks,
the generative arm's augmented decoder). The perturbation is target-agnostic, so the two axes
live in separate files:

    --config   a perturbation config (configs/experiment/foundation/robustness_*.yaml)
    --targets  the shared registry   (configs/experiment/robustness_targets.yaml)
    --target   optional: restrict to one target by name

The runner resolves (perturbation x present targets), skipping targets whose checkpoint does not
exist yet, so the suite runs incrementally as each arm is trained.

CHECKPOINT CONTRACT: every checkpoint is saved as {"model_state": ..., "model_config": ...}. A
consumer rebuilds the architecture from "model_config" and loads "model_state" — no external
model_type is needed. This is why registry targets are just (name, checkpoint/run_dir).

WHICH checkpoint: targets point at refit.pt, the model refit on the full training-subject set
with the cross-validated hyperparameters. Per-fold checkpoints (folds/fold{k}/best.pt) are for
extended analysis and are never consumed here. See docs/experiments/README.md §5.1.

The evaluation logic itself is not implemented yet (no data loader or models exist). This module
defines and enforces the interface; run_target() is the single place the real evaluation will
be filled in.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path("configs/experiment/robustness_targets.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def target_artifact(target: dict[str, Any]) -> Path:
    """The filesystem artifact whose presence means this target is trained.

    A checkpoint file for encoder/adapter targets, or a run directory for the generative arm.
    """
    if "checkpoint" in target:
        return Path(target["checkpoint"])
    if "run_dir" in target:
        return Path(target["run_dir"])
    raise ValueError(f"target {target.get('name')!r} has neither 'checkpoint' nor 'run_dir'")


def run_target(perturbation: dict[str, Any], target: dict[str, Any]) -> None:
    """Evaluate one perturbation against one present target.

    Not implemented: requires the data loader, the checkpoint-rebuild path, the perturbation
    transforms, and the metrics — none of which exist yet (Milestone 0). Reached only once a
    target's artifact is actually present on disk.
    """
    raise NotImplementedError(
        "robustness evaluation is not implemented yet; this stub only resolves the "
        "perturbation x target plan (Milestone 0)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Robustness evaluation driver.")
    parser.add_argument("--config", required=True, type=Path, help="perturbation config path")
    parser.add_argument("--targets", type=Path, default=DEFAULT_REGISTRY, help="target registry")
    parser.add_argument("--target", type=str, default=None, help="restrict to one target name")
    args = parser.parse_args()

    perturbation_cfg = load_yaml(args.config)
    perturbation = perturbation_cfg["perturbation"]
    registry = load_yaml(args.targets)
    targets: list[dict[str, Any]] = registry["targets"]

    if args.target is not None:
        targets = [t for t in targets if t["name"] == args.target]
        if not targets:
            parser.error(f"no target named {args.target!r} in {args.targets}")

    print(f"Perturbation: {perturbation['type']}  levels={perturbation.get('levels')}")
    print(f"Registry:     {args.targets}  ({len(targets)} target(s))")

    present, missing = [], []
    for target in targets:
        (present if target_artifact(target).exists() else missing).append(target)

    for target in missing:
        print(f"  skip    {target['name']:24s} (no artifact at {target_artifact(target)})")

    if not present:
        print("No targets are trained yet; nothing to evaluate. Train F1 first.")
        return 0

    for target in present:
        print(f"  run     {target['name']:24s} -> {target_artifact(target)}")
        run_target(perturbation, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Config to :class:`EncoderRunConfig` bridge for F0/F1 encoder runs.

**Consume or refuse.** Every leaf key in the config is either mapped to a run-config field,
checked against something this code guarantees, or recorded — and anything else is an error. This
is the point of the whole module. A config key that nothing reads is worse than an absent one,
because it reads as a commitment: `loss: class_weighted_cross_entropy` was declared and ignored
for weeks, and `device`, `model.type` and `dataset.normalize` were the same. Refusing unknown keys
makes that class of defect impossible rather than something an audit finds later.

The declarations below are therefore the contract, not documentation of it. ``FIELD_PATHS`` is a
1:1 map from config path to run-config field; ``CHECKED_PATHS`` are keys whose value this code can
only honour one way, so it verifies rather than accepts; ``RECORDED_PATHS`` are facts kept in the
run log without steering behaviour.

**What this script will not do.** It runs a single train/held-out split. That is F0's whole
protocol, but it is *not* F1's: F1 is 8-fold cross-validation over the training subjects with
D18-D21 selection and a refit, and none of that is written. A config naming a `split_manifest` is
refused rather than quietly run as one split, because a single-split number computed from F1's
config would look like F1 and not be it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from temporallens.evaluation.robustness import protocol_blockers
from temporallens.training.encoder import EncoderRunConfig, train_encoder

#: Config path -> ``EncoderRunConfig`` field. One entry per field the run actually takes.
FIELD_PATHS = {
    "experiment.name": "name",
    "experiment.seed": "seed",
    "dataset.data_dir": "data_dir",
    "dataset.window_size": "window_size",
    "dataset.stride": "stride",
    "dataset.input_channels": "input_channels",
    "dataset.num_classes": "num_classes",
    "dataset.normalize": "normalize",
    "dataset.max_windows_per_subject": "max_windows_per_subject",
    "dataset.debug_split.train_subjects": "train_subjects",
    "dataset.debug_split.held_out_subjects": "held_out_subjects",
    "model.type": "model_type",
    "model.hidden_dim": "hidden_dim",
    "model.embedding_dim": "embedding_dim",
    "model.dropout": "dropout",
    "training.loss": "loss",
    "training.batch_size": "batch_size",
    "training.epochs": "epochs",
    "training.learning_rate": "learning_rate",
    "training.weight_decay": "weight_decay",
    "training.num_workers": "num_workers",
    "training.device": "device",
    "training.save_checkpoint": "save_checkpoint",
    "evaluation.save_predictions": "save_predictions",
}

#: Keys this code can only honour one way, with the value it requires.
CHECKED_PATHS = {
    "dataset.name": "ninapro_db2",
    "dataset.exercise": "B",
    "dataset.debug_split.type": "subject_holdout",
    "tracking.json_logging": True,
}

#: Keys with several legal values, none of which this script may silently widen. `wandb_mode:
#: online` asserts remote logging that nothing performs, so it is refused rather than ignored.
ALLOWED_VALUE_PATHS = {"tracking.wandb_mode": ("disabled", "offline")}

#: Kept in the run log; they do not steer this script.
RECORDED_PATHS = ("experiment.mode", "protocol.status", "protocol.blocked_on")

#: Declared twice in the configs, once under `dataset` and once under `model`. Both are read, and
#: they have to agree — otherwise one of them is decoration and nobody knows which.
MIRRORED_PATHS = (
    ("dataset.input_channels", "model.input_channels"),
    ("dataset.num_classes", "model.num_classes"),
)

#: What a run computes. A config may declare a subset; anything else it asks for would be silently
#: absent from the results.
PRODUCED_METRICS = frozenset(
    {
        "accuracy",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "confusion_matrix",
        "expected_calibration_error",
        "brier_score",
        "overconfidence_error",
        "mean_confidence_when_wrong",
        "per_subject_accuracy",
        "per_subject_macro_f1",
    }
)

OPTIONAL_FIELD_PATHS = frozenset({"dataset.max_windows_per_subject"})


class ConfigError(ValueError):
    """A config this script will not run, with the reason stated."""


def leaf_paths(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten to dotted paths. A list is a leaf: subject lists and metric lists are values."""
    if isinstance(node, dict):
        flat: dict[str, Any] = {}
        for key, child in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(leaf_paths(child, path))
        return flat
    return {prefix: node}


def load_run_config(config_path: Path, output_dir: Path) -> tuple[EncoderRunConfig, dict[str, Any]]:
    """Build a run config, refusing anything this script cannot honour exactly."""
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a YAML mapping")
    leaves = leaf_paths(raw)

    if "dataset.split_manifest" in leaves:
        raise ConfigError(
            f"{config_path} names a split_manifest, so it is a reportable cross-validated run. "
            "This script runs a single train/held-out split only; F1's 8-fold harness, D18-D21 "
            "selection and the refit are not implemented. Running it as one split would produce "
            "a number that looks like F1 and is not."
        )

    if leaves.get("dataset.split") == "random_window":
        raise ConfigError(
            f"{config_path} is the random-window leakage demonstration (F2). This script splits "
            "by subject; a random window split needs its own fitting set "
            "(`train_split_covered_sample_stats`) and window assignment, neither of which is "
            "implemented. F2's number is only meaningful as the leakage contrast, so it must not "
            "be produced by a subject-split path."
        )

    known = set(FIELD_PATHS) | set(CHECKED_PATHS) | set(RECORDED_PATHS) | {"evaluation.metrics"}
    known.update(ALLOWED_VALUE_PATHS)
    known.update(mirror for _, mirror in MIRRORED_PATHS)
    unknown = sorted(set(leaves) - known)
    if unknown:
        raise ConfigError(
            f"{config_path} declares keys this script does not consume: {unknown}. Either wire "
            "them or remove them — a declared key that nothing reads reads as a commitment."
        )

    missing = sorted(
        path for path in FIELD_PATHS if path not in leaves and path not in OPTIONAL_FIELD_PATHS
    )
    if missing:
        raise ConfigError(f"{config_path} is missing required keys: {missing}")

    for path, required in CHECKED_PATHS.items():
        if path in leaves and leaves[path] != required:
            raise ConfigError(
                f"{config_path}: {path} must be {required!r} for this script, got "
                f"{leaves[path]!r}"
            )

    for path, allowed in ALLOWED_VALUE_PATHS.items():
        if path in leaves and leaves[path] not in allowed:
            raise ConfigError(
                f"{config_path}: {path} must be one of {list(allowed)}, got {leaves[path]!r}"
            )

    for source, mirror in MIRRORED_PATHS:
        if mirror in leaves and leaves[mirror] != leaves[source]:
            raise ConfigError(
                f"{config_path}: {mirror} is {leaves[mirror]!r} but {source} is "
                f"{leaves[source]!r}; they describe the same quantity and must agree"
            )

    declared = set(leaves.get("evaluation.metrics") or ())
    unproduced = sorted(declared - PRODUCED_METRICS)
    if unproduced:
        raise ConfigError(
            f"{config_path} declares metrics this run does not compute: {unproduced}. They would "
            f"be silently absent from the results. Computed: {sorted(PRODUCED_METRICS)}"
        )

    kwargs = {field: leaves[path] for path, field in FIELD_PATHS.items() if path in leaves}
    kwargs["data_dir"] = Path(str(kwargs["data_dir"]))
    kwargs["output_dir"] = output_dir
    recorded = {path: leaves[path] for path in RECORDED_PATHS if path in leaves}
    recorded.update({path: leaves[path] for path in ALLOWED_VALUE_PATHS if path in leaves})
    recorded["declared_metrics"] = sorted(declared)
    return EncoderRunConfig(**kwargs), recorded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the F0/F1 encoder from a config.")
    parser.add_argument("--config", required=True, type=Path, help="experiment config path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/runs"),
        help="run-directory root (not a protocol choice, so not a config key)",
    )
    args = parser.parse_args(argv)

    try:
        config, recorded = load_run_config(args.config, args.output_dir)
    except ConfigError as error:
        parser.error(str(error))

    # Reportability, not permission to train. The status blocks quoting numbers from this run, not
    # producing the checkpoint, so this is a loud note rather than a refusal.
    blockers = protocol_blockers(yaml.safe_load(args.config.read_text()), origin=str(args.config))
    if blockers:
        print(f"NOT REPORTABLE: {args.config} is protocol-blocked ({', '.join(blockers)})")
    if recorded.get("experiment.mode") == "debug":
        print("DEBUG RUN: a smoke test of the pipeline, never a measurement.")

    print(
        f"Training {config.name}: {len(config.train_subjects)} train subject(s), "
        f"{len(config.held_out_subjects)} held out, {config.epochs} epoch(s)"
    )
    result = train_encoder(config)
    print(f"Run directory: {result.run_dir}")
    for key in ("accuracy", "macro_f1"):
        print(f"  {key}: {result.metrics[key]:.4f}")
    print(f"  windows crossing a (label, repetition) boundary: {result.boundary_crossing_windows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

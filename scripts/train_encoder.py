#!/usr/bin/env python3
"""Config bridge for F0/F1 encoder runs: F0's single split, or F1's cross-validation harness.

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

**Two paths, chosen by the config.** A config with a ``debug_split`` is F0: one train/held-out
split through :class:`EncoderRunConfig`. A config naming a ``split_manifest`` is F1, and goes only
to :func:`~temporallens.training.subject_cv.run_subject_cross_validation`: eight folds, D18-D21
selection, D19's escalation, D48's development artifacts, D33's temperature and the refit. The F0
bridge still refuses a manifest config, because a single-split number computed from F1's config
would look like F1 and not be it.

**F1's keys come in three kinds.** ``F1_FIELD_PATHS`` map to run-config fields. ``F1_CHECKED_PATHS``
declare protocol constants the harness implements one way; each expected value is the code's own
constant where one exists, so the config and the code cannot drift apart silently.
``F1_FINAL_TEST_PATHS`` declare the CUDA-only held-out evaluation (D47), which this script never
runs and which is not implemented: they are checked and recorded so a changed declaration is still
refused, and the run summary says the final test was not evaluated.

**F1 is preliminary-only for now, on every device.** The run's scope is explicit: without
``--reportable`` it is preliminary even on CUDA, and with it the harness refuses, because the
final-test stage is not written or rehearsed (``REPORTABLE_EXECUTION_BLOCKERS``). Both scopes keep
the clean-tree guard.

**What this script will not do.** Score the eight final-test subjects. It never loads them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

from temporallens.data.splits import SplitManifest, SplitManifestError
from temporallens.evaluation.calibration import (
    INVERSE_TEMPERATURE_BOUNDS,
    NLL_NON_REGRESSION_TOLERANCE,
    SOLVER,
    SOLVER_MAX_ITERATIONS,
    SOLVER_TOLERANCE,
    TemperatureFitError,
)
from temporallens.evaluation.robustness import protocol_blockers
from temporallens.models.checkpoint import development_checkpoint_path
from temporallens.training.class_weights import CLASS_WEIGHTS_DIGEST_ENCODING
from temporallens.training.encoder import EncoderRunConfig, train_encoder
from temporallens.training.selection import (
    ESCALATED_HORIZON_EPOCHS,
    EXPECTED_FOLDS,
    EXPECTED_VALIDATION_SUBJECTS,
    INITIAL_HORIZON_EPOCHS,
    SELECTION_WINDOW_EPOCHS,
)
from temporallens.training.subject_cv import (
    ADAM_BETAS,
    ADAM_EPSILON,
    ALL_32_REFIT,
    COMMON_BUDGET_DEVELOPMENT,
    PRELIMINARY,
    REPORTABLE,
    REPORTABLE_EXECUTION_BLOCKERS,
    ArtifactExistsError,
    HorizonBlockedError,
    ReportableExecutionError,
    SubjectCVConfig,
    run_subject_cross_validation,
)
from temporallens.utils.run_logger import UncommittedCodeError
from temporallens.utils.yaml_io import DuplicateKeyError, load_yaml

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
#: `experiment.mode` decides whether a run may be reportable, and this script only trains.
ALLOWED_VALUE_PATHS = {
    "tracking.wandb_mode": ("disabled", "offline"),
    "experiment.mode": ("debug", "train"),
}

#: Checked keys that must be present. A check that applies only when the key exists is one a
#: config can dodge by leaving the key out.
REQUIRED_CHECK_PATHS = (*CHECKED_PATHS, "experiment.mode")

#: Kept in the run log; they do not steer this script.
RECORDED_PATHS = ("protocol.status", "protocol.blocked_on")

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


def read_config(config_path: Path) -> dict[str, Any]:
    """Read a config once, strictly: a repeated key is an error, never last-write-wins."""
    return read_config_with_digest(config_path)[0]


def read_config_with_digest(config_path: Path) -> tuple[dict[str, Any], str]:
    """Read a config and the SHA-256 of the exact bytes that were parsed.

    One read for both, so the recorded digest can never describe a different version of the file
    from the one the run was built from.
    """
    try:
        content = config_path.read_bytes()
        raw = load_yaml(content.decode("utf-8"))
    except DuplicateKeyError as error:
        raise ConfigError(f"{config_path}: {error}") from error
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{config_path}: cannot read config: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a YAML mapping")
    return raw, hashlib.sha256(content).hexdigest()


def load_run_config(config_path: Path, output_dir: Path) -> tuple[EncoderRunConfig, dict[str, Any]]:
    """Read and build a run config, refusing anything this script cannot honour exactly."""
    return build_run_config(read_config(config_path), config_path, output_dir)


def build_run_config(
    raw: dict[str, Any], config_path: Path, output_dir: Path
) -> tuple[EncoderRunConfig, dict[str, Any]]:
    """Build a run config from an already-read config mapping."""
    leaves = leaf_paths(raw)

    if leaves.get("dataset.split") == "random_window":
        raise ConfigError(
            f"{config_path} is the random-window leakage demonstration (F2). This script splits "
            "by subject; D44's window assignment is implemented separately, but its eight-fold "
            "training, fold-only fitting statistics, refit and temperature-calibration harness "
            "are not implemented here. F2's number is only meaningful as the leakage contrast, so it must not "
            "be produced by a subject-split path."
        )

    if "dataset.split_manifest" in leaves:
        raise ConfigError(
            f"{config_path} names a split_manifest, so it is a reportable cross-validated run. "
            "The single-split F0 path will not run it: one split would produce a number that "
            "looks like F1 and is not. Use build_f1_config, which runs the eight-fold harness."
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
        {path for path in FIELD_PATHS if path not in OPTIONAL_FIELD_PATHS}
        .union(REQUIRED_CHECK_PATHS)
        .difference(leaves)
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
    try:
        config = EncoderRunConfig(**kwargs)
    except ValueError as error:
        raise ConfigError(f"{config_path}: {error}") from error
    return config, recorded


# --- F1: the cross-validation harness ---------------------------------------------------------

#: F1 config path -> ``SubjectCVConfig`` field. The manifest path is resolved separately.
F1_FIELD_PATHS = {
    "experiment.name": "name",
    "experiment.seed": "seed",
    "dataset.data_dir": "data_dir",
    "dataset.window_size": "window_size",
    "dataset.stride": "stride",
    "dataset.input_channels": "input_channels",
    "dataset.num_classes": "num_classes",
    "model.hidden_dim": "hidden_dim",
    "model.embedding_dim": "embedding_dim",
    "model.dropout": "dropout",
    "training.batch_size": "batch_size",
    "training.learning_rate": "learning_rate",
    "training.weight_decay": "weight_decay",
    "training.num_workers": "num_workers",
    "training.device": "device",
}
F1_MANIFEST_PATH = "dataset.split_manifest"

_OPT = "training.optimization"
_SEL = "training.epoch_selection"
_ESC = "training.horizon_escalation"
_CW = "training.checkpoint_metadata.class_weights"
_DEV = "training.development_checkpoints"
_TF = "evaluation.temperature_fit"
_CE = "evaluation.calibration_execution"
_TIES = f"{_CE}.equal_mass_tie_handling"
_VTF = f"{_CE}.validation_temperature_fit"

#: Protocol constants the harness implements exactly one way (D11, D18-D21, D31, D33, D38-D40,
#: D48, D81, D85). A config may only restate them; a different value is refused.
F1_CHECKED_PATHS: dict[str, Any] = {
    "dataset.name": "ninapro_db2",
    "dataset.exercise": "B",
    "dataset.normalize": "train_subject_covered_sample_stats",
    "model.type": "cnn1d",
    "tracking.json_logging": True,
    "experiment.mode": "train",
    "training.loss": "class_weighted_cross_entropy",
    "training.epochs": INITIAL_HORIZON_EPOCHS,
    "training.save_checkpoint": True,
    f"{_OPT}.optimizer": "adam",
    f"{_OPT}.betas": list(ADAM_BETAS),
    f"{_OPT}.epsilon": ADAM_EPSILON,
    f"{_OPT}.amsgrad": False,
    f"{_OPT}.weight_decay_mode": "coupled_l2_all_trainable_parameters",
    f"{_OPT}.scheduler": "none",
    f"{_OPT}.initialization": "seeded_pytorch_module_defaults",
    f"{_OPT}.initialization_seed": "experiment.seed",
    f"{_OPT}.precision": "float32",
    f"{_OPT}.mixed_precision": False,
    f"{_OPT}.gradient_clipping": "none",
    f"{_OPT}.nonfinite_loss_or_gradients": "fail_run_before_optimizer_step",
    f"{_SEL}.metric": "validation_macro_f1",
    f"{_SEL}.validation_subject_aggregation": "equal_weight_mean",
    f"{_SEL}.expected_validation_subjects": EXPECTED_VALIDATION_SUBJECTS,
    f"{_SEL}.macro_f1_class_labels": "all_dataset_classes",
    f"{_SEL}.undefined_class_f1": 0.0,
    f"{_SEL}.smoothing": "trailing_moving_average",
    f"{_SEL}.smoothing_window_epochs": SELECTION_WINDOW_EPOCHS,
    f"{_SEL}.full_window_only": True,
    f"{_SEL}.tie_break": "earliest_epoch",
    f"{_SEL}.smoothed_score_role": "checkpoint_selection_diagnostic_only",
    f"{_SEL}.reportable_metrics_source": "selected_checkpoint_predictions",
    f"{_ESC}.trigger": "ceiling_median_selected_fold_epochs_equals_fold_horizon",
    f"{_ESC}.next_fold_epochs": ESCALATED_HORIZON_EPOCHS,
    f"{_ESC}.rerun_all_folds_from_scratch": True,
    f"{_ESC}.maximum_automatic_escalations": 1,
    f"{_ESC}.on_repeated_trigger": "block_pending_owner_decision",
    f"{_ESC}.refit_before_resolution": "forbidden",
    f"{_ESC}.test_evaluation_before_resolution": "forbidden",
    "training.refit.subject_scope": "all_manifest_training_subjects",
    "training.refit.selected_fold_count": EXPECTED_FOLDS,
    "training.refit.epoch_rule": "ceiling_median_selected_fold_epochs",
    "training.refit.initialization": "fresh_from_experiment_seed",
    "training.refit.early_stopping": False,
    f"{_CW}.field": "class_weights_sha256",
    f"{_CW}.checkpoints": [COMMON_BUDGET_DEVELOPMENT, ALL_32_REFIT],
    f"{_CW}.encoding": CLASS_WEIGHTS_DIGEST_ENCODING,
    f"{_CW}.class_order": "ascending_labels_0_through_17",
    f"{_CW}.shape": [18],
    f"{_CW}.source": "actual_d11_float32_weights_from_complete_matching_training_window_pool",
    f"{_CW}.source_window_identity": "recorded_separately_from_weight_digest",
    f"{_CW}.repetition_zero_filter": "unchanged_by_d81",
    f"{_DEV}.source": "accepted_f1_selection_trajectory_at_common_budget",
    f"{_DEV}.training_budget": "d18_d19_accepted_refit_budget",
    f"{_DEV}.accepted_horizon_only": True,
    f"{_DEV}.retain_epoch_snapshots_until_budget_freeze": True,
    f"{_DEV}.snapshot_contents": "all_model_parameters_and_buffers",
    f"{_DEV}.per_fold_best_substitution": "forbidden",
    f"{_DEV}.training_prefix_horizon_dependence": "forbidden",
    f"{_DEV}.validation_changes_training_rng": "forbidden",
    f"{_DEV}.validation_updates_parameters_or_buffers": "forbidden",
    f"{_DEV}.missing_or_incompatible_snapshot": "block_artifact_creation",
    f"{_DEV}.required_source_provenance": [
        "source_run_id",
        "fold_index",
        "source_epoch",
        "accepted_horizon",
        "source_checkpoint_sha256",
        "manifest_hash",
        "training_subject_ids",
        "scoring_subject_ids",
        "normalization_sha256",
        "class_weights_sha256",
        "config_digest",
        "training_rng_provenance",
        "device",
    ],
    f"{_TF}.prediction_source": "pooled_subject_out_of_fold_fixed_budget_models",
    f"{_TF}.fold_source": "split_manifest.folds",
    f"{_TF}.population_fit": "fold_training_complement_28_subjects",
    f"{_TF}.query_subjects": "current_fold_validation_subjects",
    f"{_TF}.query_repetitions": "all_manifest_repetitions",
    f"{_TF}.repetition_zero_rest": "excluded_from_temperature_fitting",
    f"{_TF}.encoder_checkpoint_source": "accepted_f1_selection_trajectory_at_common_budget",
    f"{_TF}.training_budget": "d18_d19_accepted_refit_budget",
    f"{_TF}.per_fold_best_checkpoint_substitution": "forbidden",
    f"{_TF}.objective": "ordinary_negative_log_likelihood",
    f"{_TF}.class_weighting": "none",
    f"{_TF}.subject_weighting": "equal",
    f"{_TF}.repeat_weighting": "equal_within_subject",
    f"{_TF}.window_weighting": "equal_within_repeat",
    f"{_TF}.median_fold_temperature_for_refit": "forbidden",
    f"{_TF}.per_fold_temperatures_role": "heterogeneity_diagnostic_only",
    f"{_TF}.selected_fold_test_temperature_source": (
        "own_outer_fold_predictions_after_checkpoint_selection"
    ),
    f"{_TF}.fit_scope": "development_only",
    f"{_TF}.freeze_before_final_test": True,
    f"{_TF}.final_test_fitting": "forbidden",
    f"{_TF}.development_metrics_role": "calibration_fitting_diagnostic_only",
    f"{_TF}.artifact": "model_config.probability_calibration",
    f"{_TF}.required_provenance": [
        "manifest_hash",
        "fold_artifact_sha256",
        "source_trajectory_provenance",
        "config_digests",
        "query_logits_sha256",
        "subject_ids",
        "repetition_ids",
        "fitting_weights",
        "fit_diagnostics",
    ],
    f"{_TF}.execution.status": "ready",
    f"{_TF}.execution.solver": SOLVER,
    f"{_TF}.execution.parameter": "inverse_temperature",
    f"{_TF}.execution.dtype": "float64",
    f"{_TF}.execution.bounds": list(INVERSE_TEMPERATURE_BOUNDS),
    f"{_TF}.execution.tolerance": SOLVER_TOLERANCE,
    f"{_TF}.execution.max_iterations": SOLVER_MAX_ITERATIONS,
    f"{_TF}.execution.nll_evaluation": "stable_logsumexp",
    f"{_TF}.execution.fitting_weights": "normalized_d33_weights",
    f"{_TF}.execution.boundary_check": "weighted_nll_derivative_at_bounds",
    f"{_TF}.execution.nll_non_regression_reference": "temperature_1",
    f"{_TF}.execution.nll_non_regression_tolerance": NLL_NON_REGRESSION_TOLERANCE,
    f"{_TF}.execution.persist_solver_diagnostics": True,
    f"{_TF}.execution.invalid_or_unresolved_boundary_fit": "block_reportable_evaluation",
    f"{_CE}.brier_reduction": "mean_windows_of_sum_over_classes",
    f"{_TIES}.quantile_method": "linear",
    f"{_TIES}.boundary_assignment": "upper_bin_for_equal_cutpoint",
    f"{_TIES}.tied_confidences": "keep_together",
    f"{_TIES}.duplicate_cutpoints": "collapse",
    f"{_TIES}.empty_bins": "omit",
    f"{_TIES}.effective_bin_count": "report",
    f"{_TIES}.bin_occupancies": "report",
    f"{_TIES}.label_information_in_boundaries": "forbidden",
    f"{_CE}.validation_temperature_source": "leave_one_subject_out_within_selected_outer_fold",
    f"{_VTF}.model_source": "same_frozen_selected_fold_checkpoint_for_all_four_fits",
    f"{_VTF}.stage": "after_checkpoint_freeze",
    f"{_VTF}.fitting_subjects": "other_three_outer_validation_subjects",
    f"{_VTF}.scoring_subject": "excluded_outer_validation_subject_only",
    f"{_VTF}.fits_per_fold": EXPECTED_VALIDATION_SUBJECTS,
    f"{_VTF}.weighting": "d33_equal_subject_repeat_and_within_repeat_window_weights",
    f"{_VTF}.repetition_zero_rest_source": f"{_TF}.repetition_zero_rest",
    f"{_VTF}.metric_aggregation": "equal_weight_mean_of_four_subject_metrics",
    f"{_VTF}.bin_recomputation": "separately_for_each_subject_and_probability_version",
    f"{_VTF}.checkpoint_selection_subject_reuse": True,
    f"{_VTF}.result_role": "development_diagnostic_conditional_on_selected_checkpoint",
    f"{_VTF}.unbiased_nested_validation_estimate": False,
    f"{_VTF}.refit_temperature_estimator_use": "forbidden",
    f"{_VTF}.persist_fitting_and_scoring_subject_ids_and_temperatures": True,
}

_SUB = "evaluation.calibration_subgroups"
#: The CUDA-only held-out evaluation (D8, D10/D83, D23, D25, D26/D79, D47, D80). Not implemented
#: and never run here; checked so a changed declaration is refused rather than silently ignored.
F1_FINAL_TEST_PATHS: dict[str, Any] = {
    "evaluation.primary_classification_metric": "macro_f1",
    "evaluation.primary_calibration_metric": "expected_calibration_error_temperature_scaled",
    "evaluation.ece.bins": 10,
    "evaluation.ece.binning": "equal_mass",
    "evaluation.ece.scope": "top_label",
    "evaluation.ece.temperature_scaling": True,
    "evaluation.ece.temperature_source": "model_config.probability_calibration",
    "evaluation.ece.report": ["raw", "temperature_scaled"],
    "evaluation.brier.report": ["raw", "temperature_scaled"],
    f"{_CE}.bootstrap_seed": "experiment.seed",
    f"{_CE}.bootstrap_resamples": 10000,
    f"{_CE}.bootstrap_confidence": 0.95,
    f"{_CE}.raw_versus_scaled_result_keys": "metric_family_plus_raw_or_temperature_scaled_suffix",
    f"{_SUB}.status": "ready",
    f"{_SUB}.grouping": "true_label",
    f"{_SUB}.subgroups": ["rest_only", "gesture_only"],
    f"{_SUB}.probabilities": "original_18_class_probabilities_and_predictions",
    f"{_SUB}.temperature": "same_frozen_population_temperature",
    f"{_SUB}.rest_class_removal_or_renormalization": "forbidden",
    f"{_SUB}.subgroup_temperature_fitting": "forbidden",
    f"{_SUB}.bins": "recomputed_within_each_subgroup_and_subject",
    f"{_SUB}.pooled_ece_decomposition": "not_a_weighted_average_of_subgroups",
    f"{_SUB}.primary_family_membership": "excluded",
    "evaluation.uncertainty_resampling_unit": "subject",
    "evaluation.metric_source": "frozen_checkpoint_predictions",
    "evaluation.requires_frozen_checkpoints": ["refit", "selected_fold_set"],
    "evaluation.metrics": [
        "accuracy",
        "macro_f1",
        "per_subject_accuracy",
        "per_class_precision",
        "per_class_recall",
        "confusion_matrix",
        "expected_calibration_error",
        "overconfidence_error",
        "mean_confidence_when_wrong",
        "per_subject_expected_calibration_error",
        "per_subject_macro_f1",
        "per_subject_brier_score",
        "per_subject_overconfidence_error",
        "brier_score",
        "rest_only_expected_calibration_error",
        "gesture_only_expected_calibration_error",
        "rest_only_brier_score",
        "gesture_only_brier_score",
    ],
    "evaluation.save_predictions": True,
}

#: Kept in the run log; they do not steer the harness.
F1_RECORDED_PATHS = ("protocol.status", "protocol.blocked_on", "tracking.wandb_mode")
F1_MIRRORED_PATHS = MIRRORED_PATHS


def _same(value: Any, expected: Any) -> bool:
    """Equality that refuses YAML's ``1`` for ``true`` and ``1`` for ``1.0``-style surprises."""
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(_same(v, e) for v, e in zip(value, expected, strict=True))
        )
    if isinstance(expected, bool) or isinstance(value, bool):
        return type(value) is type(expected) and value == expected
    if isinstance(expected, float):
        return isinstance(value, (int, float)) and float(value) == expected
    return type(value) is type(expected) and value == expected


def _expected_development_pattern(name: str) -> str:
    """The canonical D48 path the harness writes, as a config's ``checkpoint_pattern`` names it."""
    return str(development_checkpoint_path("checkpoints", name, fold=0)).replace(
        "/fold0/", "/fold{fold_index}/"
    )


def build_f1_config(
    raw: dict[str, Any], config_path: Path
) -> tuple[SubjectCVConfig, dict[str, Any]]:
    """Build F1's harness config from a manifest config, refusing anything it cannot honour."""
    leaves = leaf_paths(raw)
    if F1_MANIFEST_PATH not in leaves:
        raise ConfigError(f"{config_path} names no split_manifest, so it is not an F1 config")
    pattern_path = f"{_TF}.checkpoint_pattern"
    known = {F1_MANIFEST_PATH, pattern_path}
    known.update(F1_FIELD_PATHS, F1_CHECKED_PATHS, F1_FINAL_TEST_PATHS, F1_RECORDED_PATHS)
    known.update(mirror for _, mirror in F1_MIRRORED_PATHS)
    unknown = sorted(set(leaves) - known)
    if unknown:
        raise ConfigError(
            f"{config_path} declares keys the F1 harness does not consume: {unknown}. Either "
            "wire them or remove them — a declared key that nothing reads reads as a commitment."
        )
    missing = sorted(known.difference(F1_RECORDED_PATHS).difference(leaves))
    if missing:
        raise ConfigError(f"{config_path} is missing required keys: {missing}")
    for table in (F1_CHECKED_PATHS, F1_FINAL_TEST_PATHS):
        for path, expected in table.items():
            if not _same(leaves[path], expected):
                raise ConfigError(
                    f"{config_path}: {path} must be {expected!r} for the F1 harness, got "
                    f"{leaves[path]!r}"
                )
    if (
        leaves.get("tracking.wandb_mode", "offline")
        not in ALLOWED_VALUE_PATHS["tracking.wandb_mode"]
    ):
        raise ConfigError(f"{config_path}: tracking.wandb_mode must be disabled or offline")
    for source, mirror in F1_MIRRORED_PATHS:
        if leaves[mirror] != leaves[source]:
            raise ConfigError(f"{config_path}: {mirror} and {source} must agree")
    expected_pattern = _expected_development_pattern(str(leaves["experiment.name"]))
    if leaves[pattern_path] != expected_pattern:
        raise ConfigError(
            f"{config_path}: {pattern_path} must be {expected_pattern!r}, the path the harness "
            f"writes, got {leaves[pattern_path]!r}"
        )
    try:
        manifest = SplitManifest.load(Path(str(leaves[F1_MANIFEST_PATH])))
    except SplitManifestError as error:
        raise ConfigError(f"{config_path}: {error}") from error

    kwargs: dict[str, Any] = {field: leaves[path] for path, field in F1_FIELD_PATHS.items()}
    kwargs["data_dir"] = Path(str(kwargs["data_dir"]))
    try:
        config = SubjectCVConfig(manifest=manifest, **kwargs)
    except ValueError as error:
        raise ConfigError(f"{config_path}: {error}") from error
    recorded = {path: leaves[path] for path in F1_RECORDED_PATHS if path in leaves}
    recorded["experiment.mode"] = leaves["experiment.mode"]
    recorded["split_manifest"] = str(leaves[F1_MANIFEST_PATH])
    recorded["final_test_declarations"] = "checked_not_run"
    return config, recorded


def run_f1(
    raw: dict[str, Any],
    config_path: Path,
    config_file_sha256: str,
    *,
    output_dir: Path,
    checkpoint_root: Path,
    reportable: bool = False,
) -> int:
    """Run F1's development harness from a read config. Returns a process exit status.

    ``reportable`` requests the canonical scope explicitly; the harness refuses it before
    loading any data while :data:`REPORTABLE_EXECUTION_BLOCKERS` remains.
    """
    config, recorded = build_f1_config(raw, config_path)
    blockers = protocol_blockers(raw, origin=str(config_path))
    if blockers:
        raise ConfigError(f"{config_path} is not protocol-ready: {', '.join(blockers)}")
    metadata = {
        **recorded,
        "config_path": str(config_path),
        "config_file_sha256": config_file_sha256,
    }
    print(
        f"F1 {config.name}: {config.manifest.num_folds} folds over "
        f"{len(config.manifest.training_subjects)} development subjects; the "
        f"{len(config.manifest.test_subjects)} final-test subjects are never loaded."
    )
    try:
        result = run_subject_cross_validation(
            config,
            checkpoint_base=checkpoint_root,
            output_dir=output_dir,
            metadata=metadata,
            scope=REPORTABLE if reportable else PRELIMINARY,
            require_clean_tree=True,  # separate from the scope; both scopes keep it
        )
    except HorizonBlockedError as error:
        print(f"BLOCKED (D19): {error}")
        return 3
    except TemperatureFitError as error:
        print(f"BLOCKED (D39): the pooled refit temperature failed: {error}")
        return 4
    if result.lineage == PRELIMINARY:
        why = "; ".join(REPORTABLE_EXECUTION_BLOCKERS) or "run without --reportable"
        print(
            f"PRELIMINARY (D47), on every device ({why}). Artifacts are under "
            f"{checkpoint_root / PRELIMINARY}; report development numbers only, in their own "
            "labelled table."
        )
    print(f"Run directory: {result.run_dir}")
    print(f"  accepted horizon: {result.accepted_horizon}, common budget B: {result.budget}")
    print(f"  selected epochs: {[s.epoch for s in result.selections]}")
    print(f"  refit temperature: {result.refit_temperature.temperature:.4f}")
    print("  final-test evaluation: not run (a separate CUDA-only stage, not implemented)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the F0/F1 encoder from a config.")
    parser.add_argument("--config", required=True, type=Path, help="experiment config path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/runs"),
        help="run-directory root (not a protocol choice, so not a config key)",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints"),
        help="F1 artifact root; preliminary runs write under <root>/preliminary (D47)",
    )
    parser.add_argument(
        "--reportable",
        action="store_true",
        help="request F1's canonical CUDA run; refused until the final-test stage exists",
    )
    args = parser.parse_args(argv)

    try:
        raw, config_file_sha256 = read_config_with_digest(args.config)
        if F1_MANIFEST_PATH in leaf_paths(raw):
            return run_f1(
                raw,
                args.config,
                config_file_sha256,
                output_dir=args.output_dir,
                checkpoint_root=args.checkpoint_root,
                reportable=args.reportable,
            )
        if args.reportable:
            raise ConfigError("--reportable applies to F1's cross-validation harness only")
        config, recorded = build_run_config(raw, args.config, args.output_dir)
    except ConfigError as error:
        parser.error(str(error))
    except (UncommittedCodeError, ArtifactExistsError, ReportableExecutionError) as error:
        parser.error(str(error))

    # Reportability, not permission to train. The status blocks quoting numbers from this run, not
    # producing the checkpoint, so this is a loud note rather than a refusal.
    try:
        blockers = protocol_blockers(raw, origin=str(args.config))
    except ValueError as error:
        parser.error(str(error))
    if blockers:
        print(f"NOT REPORTABLE: {args.config} is protocol-blocked ({', '.join(blockers)})")
    if recorded.get("experiment.mode") == "debug":
        print("DEBUG RUN: a smoke test of the pipeline, never a measurement.")

    print(
        f"Training {config.name}: {len(config.train_subjects)} train subject(s), "
        f"{len(config.held_out_subjects)} held out, {config.epochs} epoch(s)"
    )
    # Only a debug run may start from uncommitted code; anything else could become a reported
    # number, and a reported number has to be traceable to a commit.
    metadata = {
        **recorded,
        "config_path": str(args.config),
        "config_file_sha256": config_file_sha256,
    }
    try:
        result = train_encoder(
            config,
            metadata=metadata,
            require_clean_tree=recorded["experiment.mode"] != "debug",
        )
    except UncommittedCodeError as error:
        parser.error(str(error))
    print(f"Run directory: {result.run_dir}")
    for key in ("accuracy", "macro_f1"):
        print(f"  {key}: {result.metrics[key]:.4f}")
    print(f"  windows crossing a (label, repetition) boundary: {result.boundary_crossing_windows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

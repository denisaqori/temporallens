"""Structural contracts linking experiment configs to manifests and specifications."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = "configs/splits/subject_independent_v1.yaml"

REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS = (
    "configs/experiment/foundation/baseline_cnn_subject_split.yaml",
    "configs/experiment/foundation/robustness_amplitude_scaling.yaml",
    "configs/experiment/foundation/robustness_channel_dropout.yaml",
    "configs/experiment/foundation/robustness_noise.yaml",
    "configs/experiment/generation/gen_calibration_efficiency.yaml",
    "configs/experiment/generation/gen_personalization_efficiency.yaml",
    "configs/experiment/generation/gen_synthetic_quality.yaml",
    "configs/experiment/generation/gen_vae_train.yaml",
    "configs/experiment/language/adapter_llama3b_subject_split.yaml",
    "configs/experiment/language/adapter_random_transformer.yaml",
    "configs/experiment/language/adapter_text_summary_only.yaml",
)

DEBUG_CONFIGS = (
    "configs/experiment/foundation/debug_tiny.yaml",
    "configs/experiment/generation/gen_vae_debug.yaml",
    "configs/experiment/language/adapter_llama1b_local.yaml",
    "configs/experiment/language/adapter_mock_debug.yaml",
)


def _load(relative_path: str) -> dict[str, Any]:
    raw = yaml.safe_load((REPO_ROOT / relative_path).read_text())
    assert isinstance(raw, dict)
    return raw


@pytest.mark.parametrize("relative_path", REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS)
def test_reportable_subject_independent_configs_use_only_the_manifest(
    relative_path: str,
) -> None:
    dataset = _load(relative_path)["dataset"]
    assert dataset["split_manifest"] == MANIFEST_PATH
    assert "split" not in dataset
    assert "held_out_subjects" not in dataset
    assert "debug_split" not in dataset


@pytest.mark.parametrize("relative_path", DEBUG_CONFIGS)
def test_debug_configs_use_an_explicit_nonreportable_subject_subset(relative_path: str) -> None:
    config = _load(relative_path)
    dataset = config["dataset"]
    assert config["experiment"]["mode"] == "debug"
    assert dataset["debug_split"]["type"] == "subject_holdout"
    assert dataset["debug_split"]["held_out_subjects"]
    assert "split_manifest" not in dataset
    assert "split" not in dataset
    assert "held_out_subjects" not in dataset


@pytest.mark.parametrize(
    "relative_path",
    (
        "configs/experiment/generation/gen_calibration_efficiency.yaml",
        "configs/experiment/generation/gen_personalization_efficiency.yaml",
    ),
)
def test_g3_g4_do_not_duplicate_manifest_repetition_partitions(relative_path: str) -> None:
    config = _load(relative_path)
    assert "eligible_repetitions" not in config["personalization"]
    assert "evaluation_repetitions" not in config["evaluation"]


def test_g2_fails_closed_without_using_final_test_subjects_as_an_iterative_gate() -> None:
    config = _load("configs/experiment/generation/gen_synthetic_quality.yaml")
    assert config["protocol"]["status"] == "blocked_pending_decisions"
    assert "evaluation.quality_metric_contract_status" in config["protocol"]["blocked_on"]
    assert config["discriminator"]["evaluate_on"] == "pending_development_validation_subjects"
    assert config["development_gate"]["status"] == "pending_decision"
    assert config["final_test_diagnostic"]["tuning_after_observation"] == "forbidden"
    assert config["evaluation"]["quality_metric_contract_status"] == "pending_decision"


def test_f1_config_encodes_approved_epoch_constants_and_gates() -> None:
    config = _load("configs/experiment/foundation/baseline_cnn_subject_split.yaml")

    training = config["training"]
    assert training["epochs"] == 30
    assert training["early_stopping"] is False

    selection = training["epoch_selection"]
    assert selection["metric"] == "validation_macro_f1"
    assert selection["validation_subject_aggregation"] == "equal_weight_mean"
    assert selection["expected_validation_subjects"] == 4
    assert selection["macro_f1_class_labels"] == "all_dataset_classes"
    assert selection["undefined_class_f1"] == 0.0
    assert selection["smoothing"] == "trailing_moving_average"
    assert selection["smoothing_window_epochs"] == 10
    assert selection["full_window_only"] is True
    assert selection["tie_break"] == "earliest_epoch"
    assert selection["smoothed_score_role"] == "checkpoint_selection_diagnostic_only"
    assert selection["reportable_metrics_source"] == "selected_checkpoint_predictions"

    escalation = training["horizon_escalation"]
    assert escalation == {
        "trigger": "ceiling_median_selected_fold_epochs_equals_fold_horizon",
        "next_fold_epochs": 60,
        "rerun_all_folds_from_scratch": True,
        "maximum_automatic_escalations": 1,
        "on_repeated_trigger": "block_pending_owner_decision",
        "refit_before_resolution": "forbidden",
        "test_evaluation_before_resolution": "forbidden",
    }

    assert training["refit"] == {
        "subject_scope": "all_manifest_training_subjects",
        "selected_fold_count": 8,
        "epoch_rule": "ceiling_median_selected_fold_epochs",
        "initialization": "fresh_from_experiment_seed",
        "early_stopping": False,
    }
    assert config["evaluation"]["metric_source"] == "frozen_checkpoint_predictions"
    assert config["evaluation"]["requires_frozen_checkpoints"] == [
        "refit",
        "selected_fold_set",
    ]


def test_g1_fails_closed_until_subject_embedding_is_operationally_defined() -> None:
    config = _load("configs/experiment/generation/gen_vae_train.yaml")
    embedding = config["generator"]["subject_embedding"]
    assert config["protocol"]["status"] == "blocked_pending_decisions"
    assert embedding["status"] == "pending_decision"
    assert embedding["estimator"] is None
    assert embedding["pseudo_calibration_schedule"] is None
    assert embedding["target_exclusion_policy"] is None


@pytest.mark.parametrize(
    "relative_path",
    (
        "configs/experiment/generation/gen_calibration_efficiency.yaml",
        "configs/experiment/generation/gen_personalization_efficiency.yaml",
    ),
)
def test_g3_g4_fail_closed_on_schedule_objective_and_replay_control(
    relative_path: str,
) -> None:
    config = _load(relative_path)
    personalization = config["personalization"]
    assert personalization["trial_selection"]["reproducibility"]["prng"] is None
    assert personalization["adaptation"]["objective"]["optimizer"] is None
    assert personalization["synthetic"]["replay_control"]["strategy"] is None


def test_g4_records_pending_headline_ece_aggregation() -> None:
    config = _load("configs/experiment/generation/gen_calibration_efficiency.yaml")
    assert config["protocol"]["status"] == "blocked_pending_decisions"
    assert "evaluation.headline_ece_aggregation_status" in config["protocol"]["blocked_on"]
    assert config["evaluation"]["headline_ece_aggregation_status"] == "pending_decision"


def test_robustness_registry_compares_both_adapted_g3_strategies() -> None:
    registry = _load("configs/experiment/robustness_targets.yaml")
    g3_targets = [target for target in registry["targets"] if target["arm"] == "generation"]
    assert {target["strategy"] for target in g3_targets} == {
        "real_adaptation",
        "real_plus_synthetic_adaptation",
    }
    assert len({tuple(target["artifact_axes"]) for target in g3_targets}) == 1


METRIC_REFERENCE_SECTIONS = (
    (
        "docs/experiments/README.md",
        "### 3.4 Metrics",
        "### 3.5 Reproducibility",
    ),
    (
        "docs/experiments/generative-arm.md",
        "## Metrics reference — the G-series keys",
        "## Robustness (D2)",
    ),
)

METRIC_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
METRIC_TABLE_ROW_PATTERN = re.compile(
    r"^\|\s*`([a-z][a-z0-9_]*)`\s*\|",
    re.MULTILINE,
)


def _section_between(relative_path: str, start_heading: str, end_heading: str) -> str:
    text = (REPO_ROOT / relative_path).read_text()
    _, start, remainder = text.partition(start_heading)
    assert start, f"{relative_path}: missing heading {start_heading!r}"
    section, end, _ = remainder.partition(end_heading)
    assert end, f"{relative_path}: missing heading {end_heading!r}"
    return section


def _registered_evaluation_metric_keys() -> set[str]:
    registrations: dict[str, list[str]] = {}

    for relative_path, start_heading, end_heading in METRIC_REFERENCE_SECTIONS:
        section = _section_between(relative_path, start_heading, end_heading)
        keys = METRIC_TABLE_ROW_PATTERN.findall(section)
        assert keys, f"{relative_path}: no metric rows under {start_heading!r}"

        for key in keys:
            registrations.setdefault(key, []).append(relative_path)

    duplicates = {key: paths for key, paths in registrations.items() if len(paths) > 1}
    assert not duplicates, f"metric keys registered more than once: {duplicates}"
    return set(registrations)


def _configured_evaluation_metric_keys() -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}

    for path in sorted((REPO_ROOT / "configs" / "experiment").rglob("*.yaml")):
        relative_path = path.relative_to(REPO_ROOT)
        config = _load(str(relative_path))
        evaluation = config.get("evaluation")
        if evaluation is None:
            continue
        assert isinstance(evaluation, dict), f"{relative_path}: evaluation must be a mapping"

        metrics = evaluation.get("metrics")
        if metrics is None:
            continue
        assert isinstance(metrics, list), f"{relative_path}: evaluation.metrics must be a list"

        for metric in metrics:
            assert isinstance(metric, str) and METRIC_KEY_PATTERN.fullmatch(
                metric
            ), f"{relative_path}: invalid evaluation metric key {metric!r}"

        assert len(metrics) == len(
            set(metrics)
        ), f"{relative_path}: duplicate evaluation.metrics entries"

        for metric in metrics:
            keys.setdefault(metric, []).append(str(relative_path))

    return keys


def test_every_evaluation_metric_is_registered_in_a_metric_reference_table() -> None:
    """Registration is exact name coverage; operational semantics may still be Pending."""
    registered = _registered_evaluation_metric_keys()
    configured = _configured_evaluation_metric_keys()
    unregistered = {
        metric: configs for metric, configs in configured.items() if metric not in registered
    }
    assert not unregistered, (
        "evaluation.metrics keys missing from the metric-reference tables: " f"{unregistered}"
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        "configs/experiment/generation/gen_vae_debug.yaml",
        "configs/experiment/generation/gen_vae_train.yaml",
    ),
)
def test_vae_configs_minimize_and_report_negative_elbo(relative_path: str) -> None:
    config = _load(relative_path)
    assert config["training"]["loss"] == "negative_elbo"
    assert "negative_elbo" in config["evaluation"]["metrics"]
    assert "elbo" not in config["evaluation"]["metrics"]
    if config["experiment"]["mode"] == "debug":
        assert {"negative_elbo_is_finite", "parameters_updated"} <= set(
            config["evaluation"]["checks"]
        )

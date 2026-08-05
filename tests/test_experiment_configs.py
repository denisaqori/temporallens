"""Experiment configs must consume the frozen split without duplicating it."""

from __future__ import annotations

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
    assert config["discriminator"]["evaluate_on"] == "pending_development_validation_subjects"
    assert config["development_gate"]["status"] == "pending_decision"
    assert config["final_test_diagnostic"]["tuning_after_observation"] == "forbidden"


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


def test_robustness_registry_compares_both_adapted_g3_strategies() -> None:
    registry = _load("configs/experiment/robustness_targets.yaml")
    g3_targets = [target for target in registry["targets"] if target["arm"] == "generation"]
    assert {target["strategy"] for target in g3_targets} == {
        "real_adaptation",
        "real_plus_synthetic_adaptation",
    }
    assert len({tuple(target["artifact_axes"]) for target in g3_targets}) == 1

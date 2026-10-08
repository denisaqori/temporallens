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

LANGUAGE_LAST_TOKEN_CONFIGS = (
    "configs/experiment/language/adapter_llama1b_local.yaml",
    "configs/experiment/language/adapter_mock_debug.yaml",
    "configs/experiment/language/adapter_llama3b_subject_split.yaml",
    "configs/experiment/language/adapter_random_transformer.yaml",
    "configs/experiment/language/adapter_text_summary_only.yaml",
)

ECE_CONFIGS = (
    "configs/experiment/foundation/baseline_cnn_random_split.yaml",
    "configs/experiment/foundation/baseline_cnn_subject_split.yaml",
    "configs/experiment/foundation/robustness_amplitude_scaling.yaml",
    "configs/experiment/foundation/robustness_channel_dropout.yaml",
    "configs/experiment/foundation/robustness_noise.yaml",
    "configs/experiment/generation/gen_calibration_efficiency.yaml",
    "configs/experiment/language/adapter_llama3b_subject_split.yaml",
    "configs/experiment/language/adapter_random_transformer.yaml",
    "configs/experiment/language/adapter_text_summary_only.yaml",
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
    assert "max_subjects" not in dataset
    assert "max_windows_per_subject" not in dataset


@pytest.mark.parametrize("relative_path", DEBUG_CONFIGS)
def test_debug_configs_use_an_explicit_nonreportable_subject_subset(relative_path: str) -> None:
    config = _load(relative_path)
    dataset = config["dataset"]
    assert config["experiment"]["mode"] == "debug"
    assert dataset["debug_split"]["type"] == "subject_holdout"
    train_subjects = dataset["debug_split"]["train_subjects"]
    held_out_subjects = dataset["debug_split"]["held_out_subjects"]
    assert train_subjects
    assert held_out_subjects
    assert set(train_subjects).isdisjoint(held_out_subjects)
    assert dataset["normalize"] == "train_subjects_global_stats"
    assert "max_subjects" not in dataset
    assert "split_manifest" not in dataset
    assert "split" not in dataset
    assert "held_out_subjects" not in dataset


@pytest.mark.parametrize("relative_path", LANGUAGE_LAST_TOKEN_CONFIGS)
def test_language_configs_define_the_classifier_head(relative_path: str) -> None:
    config = _load(relative_path)
    assert config["head"]["type"] == "mlp"
    assert config["head"]["pooling"] == "last_token"
    assert config["head"]["num_classes"] == 18


def test_language_spec_pins_the_combined_soft_prefix_mask_contract() -> None:
    readme = (REPO_ROOT / "docs/experiments/README.md").read_text()
    language_spec = (REPO_ROOT / "docs/experiments/language-arm.md").read_text()
    for text in (readme, language_spec):
        assert "combined_attention_mask" in text
        assert "position_ids" in text
        assert "mask.to(torch.int64).flip(-1).argmax(-1)" in text
    assert "cat([prefix_mask, text_attention_mask], dim=-1)" in readme


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


def test_g1_subject_embedding_contract_is_operationally_defined() -> None:
    """D27: every field the embedding needs is fixed, and none is left to a runner default."""
    config = _load("configs/experiment/generation/gen_vae_train.yaml")
    embedding = config["generator"]["subject_embedding"]

    assert embedding["estimator"] == "class_centred_mean_latent_residual"
    assert embedding["centring_reference"] == "training_subject_class_mean_latents"
    assert embedding["population_embedding"] == "learned_vector"
    probe = embedding["class_information_control"]
    assert probe["diagnostic"] == "support_class_composition_probe"
    assert probe["must_be_at_chance"] is True
    assert "status" not in embedding, "the estimator itself is decided; no pending status here"

    schedule = embedding["pseudo_calibration_schedule"]
    assert schedule["k_grid"] == [0, 1, 2, 5, 10, 17, 20, 34], "must match D14's deployment grid"
    assert schedule["k_draw"] == "uniform_per_subject_per_step"
    # Rotation is augmentation, but every pair must still span early/late repetitions so the
    # support shape matches inference and straddles the drift D13 measured.
    rotation = schedule["support_repetition_rotation"]
    assert rotation == [[1, 4], [2, 5], [3, 6]]
    assert all(hi - lo == 3 for lo, hi in rotation), "each support pair spans the session"
    assert sorted(r for pair in rotation for r in pair) == [
        1,
        2,
        3,
        4,
        5,
        6,
    ], "every repetition serves as support in exactly one rotation, and as a target in the rest"


def test_g1_target_exclusion_survives_support_rotation() -> None:
    """Rotation must not reintroduce the leak fixed support made impossible."""
    config = _load("configs/experiment/generation/gen_vae_train.yaml")
    embedding = config["generator"]["subject_embedding"]
    assert embedding["target_exclusion_policy"] == "draw_complement_of_support_repetitions"
    # Held-out subjects keep D13's fixed partition; rotation is training-subjects-only.
    assert embedding["held_out_subject_support"] == "d13_fixed_calibration_repetitions"


def test_g1_still_fails_closed_on_the_vae_objective_components() -> None:
    """D22 fixed the objective's sign and name; its components are still Pending."""
    config = _load("configs/experiment/generation/gen_vae_train.yaml")
    assert config["protocol"]["status"] == "blocked_pending_decisions"
    assert config["protocol"]["blocked_on"] == [
        "training.objective_components",
        "generator.subject_embedding.pseudo_calibration_schedule.rotation_selection",
        "generator.subject_embedding.class_information_control",
    ]
    components = config["training"]["objective_components"]
    assert components["status"] == "pending_decision"
    assert components["reconstruction_likelihood"] is None

    embedding = config["generator"]["subject_embedding"]
    # D27 chose the rotation pairs, but WHICH pair a draw uses inherits D14's PRNG convention,
    # which is itself Pending. A pair list is not a draw rule.
    rotation = embedding["pseudo_calibration_schedule"]["rotation_selection"]
    assert rotation["status"] == "pending_decision"
    assert rotation["rule"] is None and rotation["prng"] is None
    # Centring removes the class mean, not class-conditional covariance, so the probe is
    # load-bearing -- and a gate needs a target, a grouping, a score and a threshold.
    probe = config["generator"]["subject_embedding"]["class_information_control"]
    assert probe["status"] == "pending_decision"
    assert probe["grouping"] is None and probe["acceptance_threshold"] is None


@pytest.mark.parametrize(
    "relative_path",
    (
        "configs/experiment/generation/gen_calibration_efficiency.yaml",
        "configs/experiment/generation/gen_personalization_efficiency.yaml",
    ),
)
def test_g3_g4_encode_d28_d29_and_fail_closed_on_remaining_choices(
    relative_path: str,
) -> None:
    config = _load(relative_path)
    assert config["protocol"]["status"] == "blocked_pending_decisions"
    personalization = config["personalization"]
    assert personalization["rest_calibration_policy"] == "pending_decision"
    assert personalization["trial_selection"]["reproducibility"]["prng"] is None
    repeat_reproducibility = personalization["repeated_run_reproducibility"]
    assert repeat_reproducibility["status"] == "pending_decision"
    assert repeat_reproducibility["adaptation_seed_count"] is None
    assert repeat_reproducibility["adaptation_seed_derivation"] is None
    assert repeat_reproducibility["synthetic_draw_count"] is None
    assert repeat_reproducibility["synthetic_draw_seed_derivation"] is None
    assert repeat_reproducibility["repeat_axis_pairing"] is None
    assert "personalization.repeated_run_reproducibility" in config["protocol"]["blocked_on"]

    repeat_aggregation = config["evaluation"]["within_subject_repeat_aggregation"]
    assert repeat_aggregation == {
        "status": "pending_decision",
        "applicable_repeat_axes": "strategy_specific",
        "point_summary": None,
        "spread_summary": None,
        "axis_order_or_joint_reduction": None,
        "shared_population_reference": "deduplicate_before_summary",
        "reused_evaluation_windows_are_independent": False,
    }
    assert "evaluation.within_subject_repeat_aggregation" in config["protocol"]["blocked_on"]

    expected_strategies = {
        "population_no_adaptation",
        "real_adaptation",
        "real_plus_population_synthetic_adaptation",
        "real_plus_subject_synthetic_adaptation",
    }
    assert len(personalization["calibration_strategy"]) == len(expected_strategies)
    assert set(personalization["calibration_strategy"]) == expected_strategies

    synthetic = personalization["synthetic"]
    conditioning = synthetic["conditioning_by_strategy"]
    population = conditioning["real_plus_population_synthetic_adaptation"]
    subject = conditioning["real_plus_subject_synthetic_adaptation"]
    assert population == {
        "at_k0": "population_embedding",
        "at_k_gt_0": "population_embedding",
    }
    assert subject == {
        "at_k0": "population_embedding",
        "at_k_gt_0": "calibration_derived_from_k_only",
    }
    assert synthetic["bank_lifetime"] == {
        "at_k0": "once_per_synthetic_draw_id_shared_across_subjects_and_schedules",
        "at_k_gt_0": "once_per_subject_schedule_k_and_synthetic_draw_id",
    }
    assert synthetic["reuse_bank_across_optimizer_steps"] is True
    assert synthetic["persist_synthetic_draw_id"] is True

    replay = synthetic["replay_control"]
    assert replay["strategy"] == "paired_population_and_subject_conditioned_arms"
    assert replay["k0_shared_head_scope"] == (
        "one_per_adaptation_seed_and_synthetic_draw_id_across_subjects_and_schedules"
    )
    assert set(replay["shared_across_the_two_replay_arms"]) == {
        "synthetic_class_labels",
        "latent_noise_draws",
        "synthetic_draw_id",
        "real_calibration_windows",
        "head_initialization",
        "example_ordering",
        "adaptation_seed",
    }

    aliases = personalization["k0_execution_aliases"]
    assert len(set(aliases.values())) == 2
    assert aliases["population_no_adaptation"] == aliases["real_adaptation"]
    assert (
        aliases["real_plus_population_synthetic_adaptation"]
        == aliases["real_plus_subject_synthetic_adaptation"]
    )
    scope = personalization["k0_artifact_scope"]
    assert scope["shared_no_adaptation_head"] == {
        "repeat_identity": "none",
        "shared_across_subjects_and_schedules": True,
        "evaluation_scope": "each_held_out_subject_once",
    }
    assert scope["shared_population_conditioned_replay_head"] == {
        "repeat_identity": ["adaptation_seed", "synthetic_draw_id"],
        "shared_across_subjects_and_schedules": True,
        "evaluation_scope": "each_held_out_subject_once_per_repeat",
    }

    objective = personalization["adaptation"]["objective"]
    assert objective["loss"] == "additive_weighted_real_plus_unweighted_synthetic"
    assert objective["real_loss"] == "weight_normalised_class_weighted_cross_entropy"
    assert objective["synthetic_loss"] == "unweighted_mean_cross_entropy"
    assert objective["adam_weight_decay"] == 0.0
    development = objective["development_selection"]
    assert development["result_role"] == "hyperparameter_selection_only"
    assert development["split_manifest"] == config["dataset"]["split_manifest"]
    assert development["fold_index_source"] == "split_manifest.folds[].index"
    assert development["selection_k_values"] == [
        k for k in personalization["calibration_trials"] if k > 0
    ]
    assert development["population_fit"] == "fold_training_complement_28_subjects"
    assert development["scoring_subjects"] == "fold_validation_4_subjects"
    assert (
        development["training_subject_rule"]
        == "split_manifest.training_subjects_minus_current_fold_validation_subjects"
    )
    assert development["scoring_subject_rule"] == "current_fold_validation_subjects"
    assert development["checkpoint_selection_on_outer_four"] == "forbidden"
    artifacts = development["artifact_resolution"]
    assert artifacts["source"] == "build_from_frozen_source_configs_and_persist"
    assert (
        artifacts["encoder_source_config"]
        == "configs/experiment/foundation/baseline_cnn_subject_split.yaml"
    )
    assert (
        artifacts["generator_source_config"] == "configs/experiment/generation/gen_vae_train.yaml"
    )
    assert artifacts["generator_encoder_source"] == "fold_local_fixed_budget_encoder"
    assert artifacts["all_32_refit_reuse"] == "forbidden"
    for key in ("encoder_checkpoint_pattern", "generator_checkpoint_pattern"):
        assert "development/fold{fold_index}/fixed_budget.pt" in artifacts[key]
        assert artifacts[key] not in config["checkpoint"].values()
    assert set(artifacts["fold_fitted_components"]) == {
        "normalization_statistics",
        "class_weights",
        "encoder_and_population_head",
        "latent_centroids",
        "population_embedding",
        "generator",
    }
    assert set(artifacts["required_provenance"]) == {
        "manifest_hash",
        "fold_index",
        "training_subject_ids",
        "scoring_subject_ids",
        "component_config_digests",
        "training_budget",
    }

    k0_steps = personalization["adaptation"]["training_budget"]["synthetic_only_k0_optimizer_steps"]
    assert k0_steps == {"status": "pending_decision", "assignment_rule": None}
    assert (
        "personalization.adaptation.training_budget.synthetic_only_k0_optimizer_steps"
        in config["protocol"]["blocked_on"]
    )

    # D28/D29 settled the objective form and replay control. The numeric grid remains open, and a
    # grid declared after results are inspected is not a protocol.
    grid = objective["candidate_grid"]
    assert grid["status"] == "pending_decision"
    assert grid["learning_rate"] is None and grid["l2_sp_beta"] is None
    assert grid["optimizer_steps_scope"] is None
    assert grid["selection_aggregation"] is None


def test_g3_encodes_d30_subgroup_reporting() -> None:
    config = _load("configs/experiment/generation/gen_personalization_efficiency.yaml")
    subgroup = config["evaluation"]["subgroup_aggregation"]
    assert subgroup == {
        "aggregation": "mean_per_class_recall_within_subgroup",
        "empty_group_result": "null_metric",
        "seen_group_is_na_at": "k_equals_0",
        "unseen_group_is_na_from": "k_equals_17",
        "within_subject_summary_precedes_paired_comparison": True,
    }
    assert {
        "calibration_seen_gesture_macro_recall",
        "calibration_unseen_gesture_macro_recall",
        "rest_accuracy",
    } <= set(config["evaluation"]["metrics"])
    assert "calibration_seen_gesture_macro_accuracy" not in config["evaluation"]["metrics"]
    assert "calibration_unseen_gesture_macro_accuracy" not in config["evaluation"]["metrics"]


def test_g3_g4_share_the_same_development_selection_contract() -> None:
    g3 = _load("configs/experiment/generation/gen_personalization_efficiency.yaml")
    g4 = _load("configs/experiment/generation/gen_calibration_efficiency.yaml")
    g3_selection = g3["personalization"]["adaptation"]["objective"]["development_selection"]
    g4_selection = g4["personalization"]["adaptation"]["objective"]["development_selection"]
    assert g3_selection == g4_selection


def test_g4_records_pending_headline_ece_aggregation() -> None:
    config = _load("configs/experiment/generation/gen_calibration_efficiency.yaml")
    assert config["protocol"]["status"] == "blocked_pending_decisions"
    assert "evaluation.headline_ece_aggregation_status" in config["protocol"]["blocked_on"]
    assert config["evaluation"]["headline_ece_aggregation_status"] == "pending_decision"


@pytest.mark.parametrize("relative_path", ECE_CONFIGS)
def test_d23_ece_contract_is_present_wherever_ece_is_reported(relative_path: str) -> None:
    evaluation = _load(relative_path)["evaluation"]
    assert evaluation["ece"]["bins"] == 10
    assert evaluation["ece"]["binning"] == "equal_mass"
    assert evaluation["ece"]["scope"] == "top_label"
    assert evaluation["ece"]["temperature_scaling"] is True
    assert evaluation["ece"]["report"] == ["raw", "temperature_scaled"]
    assert evaluation["uncertainty_resampling_unit"] == "subject"
    assert "expected_calibration_error" in evaluation["metrics"]
    assert "per_subject_expected_calibration_error" in evaluation["metrics"]
    assert "brier_score" in evaluation["metrics"]


@pytest.mark.parametrize(
    "relative_path",
    (
        "configs/experiment/foundation/robustness_amplitude_scaling.yaml",
        "configs/experiment/foundation/robustness_channel_dropout.yaml",
        "configs/experiment/foundation/robustness_noise.yaml",
    ),
)
def test_robustness_uses_the_frozen_target_temperature(relative_path: str) -> None:
    config = _load(relative_path)
    assert config["evaluation"]["ece"]["temperature_source"] == "checkpoint_model_config"


def test_robustness_registry_compares_all_three_adapted_g3_strategies() -> None:
    registry = _load("configs/experiment/robustness_targets.yaml")
    all_names = [target["name"] for target in registry["targets"]]
    assert len(all_names) == len(set(all_names)), "target names must be unique"
    assert all(
        (REPO_ROOT / target["source_config"]).is_file() for target in registry["targets"]
    ), "every target must identify a versioned source experiment config"
    g3_targets = [target for target in registry["targets"] if target["arm"] == "generation"]
    assert len(g3_targets) == 3
    assert {target["strategy"] for target in g3_targets} == {
        "real_adaptation",
        "real_plus_population_synthetic_adaptation",
        "real_plus_subject_synthetic_adaptation",
    }, "D29: without the population-conditioned slice, a surviving advantage is unattributable"
    by_name = {target["name"]: target for target in g3_targets}
    assert by_name["g3_real_adaptation"]["artifact_axes"] == [
        "held_out_subject",
        "schedule_index",
        "calibration_trials",
        "adaptation_seed",
    ]
    assert by_name["g3_real_adaptation"]["k0_artifact_scope"] == {
        "model_artifact": "one_global_shared_no_adaptation_head",
        "evaluation_scope": "each_held_out_subject_once",
    }
    assert by_name["g3_real_adaptation"]["shared_k0_with"] == "f1_encoder"
    replay_axes = [
        "held_out_subject",
        "schedule_index",
        "calibration_trials",
        "adaptation_seed",
        "synthetic_draw_id",
    ]
    assert by_name["g3_population_synthetic_adaptation"]["artifact_axes"] == replay_axes
    assert by_name["g3_subject_synthetic_adaptation"]["artifact_axes"] == replay_axes
    assert {target["within_subject_repeat_aggregation"] for target in g3_targets} == {
        "pending_decision"
    }
    assert by_name["g3_population_synthetic_adaptation"]["k0_artifact_scope"] == {
        "model_artifact": (
            "one_per_adaptation_seed_and_synthetic_draw_id_shared_across_subjects_and_schedules"
        ),
        "evaluation_scope": "each_held_out_subject_once_per_repeat",
    }
    assert {target["repeat_axis_source"] for target in g3_targets} == {
        "gen_personalization_efficiency.personalization.repeated_run_reproducibility"
    }
    assert {target["source_config"] for target in g3_targets} == {
        "configs/experiment/generation/gen_personalization_efficiency.yaml"
    }
    assert (
        by_name["g3_subject_synthetic_adaptation"]["shared_k0_with"]
        == "g3_population_synthetic_adaptation"
    )
    target_names = set(all_names)
    assert all(
        target.get("shared_k0_with") in target_names
        for target in registry["targets"]
        if "shared_k0_with" in target
    )


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


def test_shared_metric_registry_is_a_three_column_markdown_table() -> None:
    section = _section_between(
        "docs/experiments/README.md",
        "### 3.4 Metrics",
        "### 3.5 Reproducibility",
    )
    table = next(block for block in section.split("\n\n") if block.startswith("| Metric |"))
    rows = table.splitlines()
    assert rows[0].split("|")[1:-1] == [" Metric ", " What it captures ", " Why it is here "]
    assert all(len(row.split("|")[1:-1]) == 3 for row in rows)


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


def test_every_blocked_on_path_resolves_to_an_unresolved_field() -> None:
    """A `blocked_on` entry must name a field that exists and is actually still open.

    Two silent failures this closes. A path naming a field that does not exist reports a blocker
    nobody can act on, and reads as protection while protecting nothing — this file has carried
    such a path before. A path naming a field that has since been *resolved* keeps a finished
    experiment blocked, which trains the owner to override the gate.
    """
    for config_path in sorted(REPO_ROOT.glob("configs/experiment/**/*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            continue
        protocol = config.get("protocol")
        if not isinstance(protocol, dict):
            continue
        relative = config_path.relative_to(REPO_ROOT)
        for path in protocol.get("blocked_on") or []:
            node: Any = config
            for part in str(path).split("."):
                assert (
                    isinstance(node, dict) and part in node
                ), f"{relative}: protocol.blocked_on names {path!r}, but {part!r} does not exist"
                node = node[part]
            if isinstance(node, dict):
                open_values = [
                    value
                    for value in node.values()
                    if value is None or (isinstance(value, str) and value.startswith("pending_"))
                ]
                assert open_values, (
                    f"{relative}: protocol.blocked_on names {path!r}, but nothing under it is "
                    f"still open (no null and no pending_* sentinel)"
                )
            else:
                assert node is None or (isinstance(node, str) and node.startswith("pending_")), (
                    f"{relative}: protocol.blocked_on names {path!r}, but its value {node!r} is "
                    f"already resolved"
                )

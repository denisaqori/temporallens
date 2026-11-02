"""Structural contracts linking experiment configs to manifests and specifications."""

from __future__ import annotations

import re
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from temporallens.evaluation.robustness import (
    pending_field_paths,
    protocol_blockers,
    source_protocol_blockers,
)
from temporallens.utils.randomness import derive_seed

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


def test_all_experiment_root_seeds_remain_42_under_d43() -> None:
    for path in sorted((REPO_ROOT / "configs" / "experiment").glob("*/*.yaml")):
        assert _load(str(path.relative_to(REPO_ROOT)))["experiment"]["seed"] == 42


def test_g3_g4_share_d43_acquisition_identity_and_realized_assignment_contract() -> None:
    from temporallens.utils.randomness import assignment_rng, derive_seed

    configs = [
        _load(f"configs/experiment/generation/{name}.yaml")
        for name in ("gen_personalization_efficiency", "gen_calibration_efficiency")
    ]
    policies = [config["personalization"]["trial_selection"] for config in configs]
    for config in configs:
        assert config["experiment"]["seed"] == 42
        assert config["personalization"]["schedules_per_subject"] == 5
    assert policies[0] == policies[1]
    policy = policies[0]
    assert policy["seed_derivation"] == "temporallens_rng_v1"
    assert policy["randomization_scope"] == "g3_g4_personalization"
    assert policy["randomization_purpose"] == "acquisition_schedule"
    assert policy["seed_indices"] == ["subject_id", "schedule_index"]
    assert policy["shared_across_strategies"] is True
    assert policy["reproducibility"]["persist_selected_trial_ids"] is True
    assert policy["reproducibility"]["persist_rng_provenance"] is True
    assert policy["reproducibility"]["persist_before_fitting"] is True
    assert policy["reproducibility"]["require_trusted_file_sha256"] is True
    assert policy["reproducibility"]["assignment_reuse"] == "load_verified_existing_or_create_once"
    assert policy["reproducibility"]["assignment_artifact"].startswith(
        "data/processed/assignments/g3_g4_personalization/v1/"
    )
    # An implementation cannot key by literal G3/G4 names, strategy or k and pass this identity.
    indices = {"subject_id": 5, "schedule_index": 0}
    seeds = [
        derive_seed(
            c["experiment"]["seed"], p["randomization_scope"], p["randomization_purpose"], indices
        )
        for c, p in zip(configs, policies, strict=True)
    ]
    assert seeds[0] == seeds[1]
    streams = [
        assignment_rng(
            c["experiment"]["seed"], p["randomization_scope"], p["randomization_purpose"], indices
        )
        .integers(0, 2**32, size=8)
        .tolist()
        for c, p in zip(configs, policies, strict=True)
    ]
    assert streams[0] == streams[1]


F2_CONFIG = "configs/experiment/foundation/baseline_cnn_random_split.yaml"
G3_CONFIGS = tuple(
    f"configs/experiment/generation/{name}.yaml"
    for name in ("gen_personalization_efficiency", "gen_calibration_efficiency")
)


def test_assignment_configs_match_the_policy_the_builders_execute() -> None:
    from temporallens.data.assignments import (
        ASSIGNMENT_REGISTRY,
        check_f2_config,
        check_g3_config,
    )

    check_f2_config(_load(F2_CONFIG))
    for relative_path in G3_CONFIGS:
        check_g3_config(_load(relative_path))
    assert (REPO_ROOT / ASSIGNMENT_REGISTRY).is_file()


def _policy_cases() -> list[tuple[str, str]]:
    from temporallens.data.assignments import F2_CONFIG_POLICY, G3_CONFIG_POLICY

    return [(F2_CONFIG, path) for path in F2_CONFIG_POLICY] + [
        (relative_path, path) for relative_path in G3_CONFIGS for path in G3_CONFIG_POLICY
    ]


@pytest.mark.parametrize(("relative_path", "leaf"), _policy_cases())
def test_assignment_policy_refuses_every_edited_or_missing_leaf(
    relative_path: str, leaf: str
) -> None:
    from temporallens.data.assignments import check_f2_config, check_g3_config

    check = check_f2_config if relative_path == F2_CONFIG else check_g3_config
    *parents, name = leaf.split(".")
    for mutate in ("edit", "delete"):
        config = _load(relative_path)
        node = config
        for part in parents:
            node = node[part]
        if mutate == "edit":
            node[name] = "edited-by-test"
        else:
            del node[name]
        with pytest.raises(ValueError, match=re.escape(leaf)):
            check(config)


def test_d43_assignments_require_pool_digest_material_assignment_and_rng_provenance() -> None:
    g1 = _load("configs/experiment/generation/gen_vae_train.yaml")
    f2 = _load("configs/experiment/foundation/baseline_cnn_random_split.yaml")
    policies = [
        g1["generator"]["subject_embedding"]["pseudo_calibration_schedule"]["reproducibility"],
        f2["evaluation"]["temperature_fit"]["execution"],
        *(
            _load(f"configs/experiment/generation/{name}.yaml")["personalization"][
                "trial_selection"
            ]["reproducibility"]
            for name in ("gen_personalization_efficiency", "gen_calibration_efficiency")
        ),
    ]
    for policy in policies:
        assert policy["persist_before_fitting"] is True
        assert policy["required_assignment_provenance"] == [
            "source_pool_identity",
            "source_pool_sha256",
            "assignment_sha256",
        ]
        assert policy["required_rng_provenance"] == [
            "key",
            "derived_seed",
            "assignment_prng",
            "numpy_version",
        ]


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
    assert dataset["normalize"] == "train_subject_covered_sample_stats"
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
    assert config["discriminator"]["evaluate_on"] == "split_manifest_fold_validation_subjects"
    assert config["development_gate"]["status"] == "pending_decision"
    assert config["final_test_diagnostic"]["tuning_after_observation"] == "forbidden"
    assert config["evaluation"]["quality_metric_contract_status"] == "pending_decision"

    artifacts = config["development_gate"]["artifact_resolution"]
    g3 = _load(artifacts["contract_source_config"])
    declared = g3["personalization"]["adaptation"]["objective"]["development_selection"][
        "artifact_resolution"
    ]
    for key in ("encoder_checkpoint_pattern", "generator_checkpoint_pattern"):
        assert artifacts[key] == declared[key]
        assert artifacts[key] not in config["checkpoint"].values()
    assert artifacts["all_32_refit_reuse"] == "forbidden"
    assert artifacts["generator_parent_encoder_identity"] == "exact_artifact_sha256"
    assert artifacts["cross_fold_latent_pooling"] == "forbidden"
    assert artifacts["nearest_neighbour_reference"] == "corresponding_fold_real_training_latents"
    assert artifacts["additional_encoder_generator_fit_set"] == "none_beyond_shared_d28_artifacts"


def test_f1_config_encodes_approved_epoch_constants_and_gates() -> None:
    config = _load("configs/experiment/foundation/baseline_cnn_subject_split.yaml")

    training = config["training"]
    assert training["epochs"] == 30
    assert training["learning_rate"] == 0.0003
    assert training["weight_decay"] == 0.0001
    assert training["optimization"] == {
        "optimizer": "adam",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "weight_decay_mode": "coupled_l2_all_trainable_parameters",
        "scheduler": "none",
        "initialization": "seeded_pytorch_module_defaults",
        "initialization_seed": "experiment.seed",
        "precision": "float32",
        "mixed_precision": False,
        "gradient_clipping": "none",
        "nonfinite_loss_or_gradients": "fail_run_before_optimizer_step",
    }
    assert (
        "early_stopping" not in training
    ), "D18/D19 own the horizon; a config key restating it can only drift from it"

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
        "generator.subject_embedding.pseudo_calibration_schedule.reproducibility",
        "generator.subject_embedding.pseudo_calibration_schedule.rotation_selection",
        "generator.subject_embedding.class_information_control",
    ]
    components = config["training"]["objective_components"]
    assert components["status"] == "pending_decision"
    assert components["reconstruction_likelihood"] is None

    embedding = config["generator"]["subject_embedding"]
    # D43 fixes the backend/factory, not WHICH pair a draw uses or its component identity.
    reproducibility = embedding["pseudo_calibration_schedule"]["reproducibility"]
    assert reproducibility["status"] == "pending_decision"
    assert reproducibility["prng"] == "numpy_pcg64"
    assert reproducibility["seed_derivation"] == "temporallens_rng_v1"
    assert reproducibility["draw_index_base"] == 0
    assert reproducibility["component_stream_keys"] is None
    assert reproducibility["persist_lossless_draw_ledger"] is True
    assert reproducibility["persist_rng_provenance"] is True
    rotation = embedding["pseudo_calibration_schedule"]["rotation_selection"]
    assert rotation["status"] == "pending_decision"
    assert rotation["rule"] is None
    assert rotation["component_stream_keys"] is None
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
    reproducibility = personalization["trial_selection"]["reproducibility"]
    assert reproducibility["status"] == "ready"
    assert reproducibility["prng"] == "numpy_pcg64"
    assert reproducibility["schedule_index_base"] == 0
    assert reproducibility["permutation_algorithm"] == "d45_three_sequential_numpy_permutations"
    assert reproducibility["permutation_call_order"] == [
        "ascending_active_gestures",
        "ascending_complement_rep_1_gestures",
        "ascending_complement_rep_4_gestures",
    ]
    assert reproducibility["second_pass_order"] == (
        "shuffled_complement_groups_interleaved_underrepresented_repetition_first"
    )
    assert reproducibility["parity_to_starting_repetition"] == {"even": 1, "odd": 4}
    assert "personalization.trial_selection.reproducibility" not in config["protocol"]["blocked_on"]
    repeat_reproducibility = personalization["repeated_run_reproducibility"]
    assert repeat_reproducibility["status"] == "pending_decision"
    assert repeat_reproducibility["adaptation_seed_count"] is None
    assert repeat_reproducibility["adaptation_seed_derivation"] == "temporallens_rng_v1"
    assert repeat_reproducibility["synthetic_draw_count"] is None
    assert repeat_reproducibility["synthetic_draw_seed_derivation"] == "temporallens_rng_v1"
    assert repeat_reproducibility["seed_index_base"] == 0
    assert repeat_reproducibility["draw_index_base"] == 0
    assert repeat_reproducibility["component_stream_keys"] is None
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
    assert development["per_fold_best_checkpoint_selection_on_outer_four"] == "forbidden"
    artifacts = development["artifact_resolution"]
    assert artifacts["source"] == "reuse_f1_fold_snapshot_and_build_fold_generator"
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
        "generator_training_budget",
        "artifact_sha256",
        "parent_encoder_sha256",
    }
    assert (
        development["generator_training_budget"] == "frozen_generator_source_config_training_epochs"
    )
    assert artifacts["generator_parent_encoder_identity"] == "exact_artifact_sha256"
    assert artifacts["cross_fold_latent_pooling"] == "forbidden"

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


def test_g3_headline_uses_macro_f1_without_dropping_unseen_classes() -> None:
    evaluation = _load("configs/experiment/generation/gen_personalization_efficiency.yaml")[
        "evaluation"
    ]
    assert evaluation["primary_classification_metric"] == "macro_f1"
    assert evaluation["headline_curve"] == "macro_f1_vs_k_curve"
    assert evaluation["macro_f1_class_labels"] == "all_dataset_classes"
    assert evaluation["undefined_class_f1"] == 0.0
    assert {"macro_f1_vs_k_curve", "accuracy_vs_k_curve"} <= set(evaluation["metrics"])
    assert evaluation["headline_extraction"] == "pending_decision"


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
    assert config["evaluation"]["headline_ece_aggregation"] == "pending_decision"


@pytest.mark.parametrize("relative_path", ECE_CONFIGS)
def test_d23_ece_contract_is_present_wherever_ece_is_reported(relative_path: str) -> None:
    evaluation = _load(relative_path)["evaluation"]
    assert evaluation["ece"]["bins"] == 10
    assert evaluation["ece"]["binning"] == "equal_mass"
    assert evaluation["ece"]["scope"] == "top_label"
    assert evaluation["ece"]["temperature_scaling"] is True
    assert evaluation["ece"]["report"] == ["raw", "temperature_scaled"]
    assert (
        evaluation["primary_calibration_metric"] == "expected_calibration_error_temperature_scaled"
    )
    assert evaluation["brier"]["report"] == ["raw", "temperature_scaled"]
    execution = evaluation["calibration_execution"]
    assert execution["raw_versus_scaled_result_keys"] == (
        "metric_family_plus_raw_or_temperature_scaled_suffix"
    )
    assert execution["bootstrap_seed"] == "experiment.seed"
    assert execution["bootstrap_resamples"] == 10000
    assert execution["bootstrap_confidence"] == 0.95
    assert execution["brier_reduction"] == "mean_windows_of_sum_over_classes"
    assert execution["equal_mass_tie_handling"] == {
        "quantile_method": "linear",
        "boundary_assignment": "upper_bin_for_equal_cutpoint",
        "tied_confidences": "keep_together",
        "duplicate_cutpoints": "collapse",
        "empty_bins": "omit",
        "effective_bin_count": "report",
        "bin_occupancies": "report",
        "label_information_in_boundaries": "forbidden",
    }
    assert "evaluation.calibration_execution" not in _load(relative_path)["protocol"]["blocked_on"]
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
    assert config["evaluation"]["ece"]["temperature_source"] == "frozen_target_calibration_artifact"


def test_f1_temperature_uses_fixed_budget_oof_models_without_median_estimator() -> None:
    config = _load("configs/experiment/foundation/baseline_cnn_subject_split.yaml")
    fit = config["evaluation"]["temperature_fit"]
    g3 = _load("configs/experiment/generation/gen_personalization_efficiency.yaml")
    development = g3["personalization"]["adaptation"]["objective"]["development_selection"]
    assert (
        fit["checkpoint_pattern"]
        == development["artifact_resolution"]["encoder_checkpoint_pattern"]
    )
    assert fit["prediction_source"] == "pooled_subject_out_of_fold_fixed_budget_models"
    assert fit["query_repetitions"] == "all_manifest_repetitions"
    assert fit["per_fold_best_checkpoint_substitution"] == "forbidden"
    assert fit["median_fold_temperature_for_refit"] == "forbidden"
    assert fit["selected_fold_test_temperature_source"] == (
        "own_outer_fold_predictions_after_checkpoint_selection"
    )
    execution = config["evaluation"]["calibration_execution"]
    assert execution["validation_temperature_source"] == (
        "leave_one_subject_out_within_selected_outer_fold"
    )
    assert execution["validation_temperature_fit"] == {
        "model_source": "same_frozen_selected_fold_checkpoint_for_all_four_fits",
        "stage": "after_checkpoint_freeze",
        "fitting_subjects": "other_three_outer_validation_subjects",
        "scoring_subject": "excluded_outer_validation_subject_only",
        "fits_per_fold": 4,
        "weighting": "d33_equal_subject_repeat_and_within_repeat_window_weights",
        "metric_aggregation": "equal_weight_mean_of_four_subject_metrics",
        "bin_recomputation": "separately_for_each_subject_and_probability_version",
        "checkpoint_selection_subject_reuse": True,
        "result_role": "development_diagnostic_conditional_on_selected_checkpoint",
        "unbiased_nested_validation_estimate": False,
        "refit_temperature_estimator_use": "forbidden",
        "persist_fitting_and_scoring_subject_ids_and_temperatures": True,
    }
    assert fit["final_test_fitting"] == "forbidden"


def test_f1_common_budget_snapshots_have_explicit_reuse_guards_and_source_identity() -> None:
    """D48 changes construction, never F1's selection, refit or device boundaries."""
    config = _load("configs/experiment/foundation/baseline_cnn_subject_split.yaml")
    policy = config["training"]["development_checkpoints"]
    assert policy["source"] == "accepted_f1_selection_trajectory_at_common_budget"
    assert policy["training_budget"] == "d18_d19_accepted_refit_budget"
    assert policy["accepted_horizon_only"] is True
    assert policy["retain_epoch_snapshots_until_budget_freeze"] is True
    assert policy["snapshot_contents"] == "all_model_parameters_and_buffers"
    for field in (
        "per_fold_best_substitution",
        "training_prefix_horizon_dependence",
        "validation_changes_training_rng",
        "validation_updates_parameters_or_buffers",
    ):
        assert policy[field] == "forbidden"
    assert policy["missing_or_incompatible_snapshot"] == "block_artifact_creation"
    assert set(policy["required_source_provenance"]) == {
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
    }
    fit = config["evaluation"]["temperature_fit"]
    assert fit["encoder_checkpoint_source"] == policy["source"]
    assert "source_trajectory_provenance" in fit["required_provenance"]
    assert config["training"]["refit"]["epoch_rule"] == "ceiling_median_selected_fold_epochs"
    assert config["training"]["horizon_escalation"]["rerun_all_folds_from_scratch"] is True


def test_f1_protocol_has_no_pending_optimizer_or_calibration_choices() -> None:
    """Protocol readiness is explicit; the separate loader test still refuses the absent harness."""
    config = _load("configs/experiment/foundation/baseline_cnn_subject_split.yaml")
    assert config["protocol"] == {"status": "ready", "blocked_on": []}

    def assert_resolved(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                assert_resolved(value)
        elif isinstance(node, list):
            for value in node:
                assert_resolved(value)
        else:
            assert node is not None
            assert not (isinstance(node, str) and node.startswith("pending_"))

    assert_resolved(config["training"]["optimization"])
    assert_resolved(config["evaluation"]["calibration_execution"])
    assert_resolved(config["evaluation"]["temperature_fit"])


def test_f2_temperature_preserves_outer_test_and_intentional_window_split() -> None:
    config = _load("configs/experiment/foundation/baseline_cnn_random_split.yaml")
    fit = config["evaluation"]["temperature_fit"]
    assert config["dataset"]["test_fraction"] == 0.2
    assert fit["folds"] == 8
    assert fit["partition_unit"] == "random_window"
    assert fit["development_pool"] == "non_test_80_percent_windows"
    assert fit["final_test_pool"] == "fixed_outer_20_percent_windows"
    assert fit["preprocessing_fit"] == "current_inner_fold_training_windows_only"
    assert fit["class_weight_fit"] == "current_inner_fold_training_windows_only"
    assert fit["persist_outer_test_and_inner_fold_assignments"] is True
    assert fit["final_test_fitting"] == "forbidden"
    assert fit["execution"]["fold_assignment_algorithm"] == "d44_global_window_split_v1"
    assert fit["execution"]["canonical_window_order"] == ["subject_id", "start_sample"]
    assert fit["execution"]["outer_test_count"] == "ceil_n_windows_divided_by_5"
    assert fit["execution"]["inner_input_order"] == "recanonicalized_non_test_windows"
    assert fit["execution"]["inner_membership"] == (
        "one_independent_permutation_then_numpy_array_split_8"
    )
    assert fit["execution"]["stratification"] == "none"
    assert fit["execution"]["composition_triggered_redraw"] == "forbidden"
    assert fit["execution"]["seed_derivation"] == "temporallens_rng_v1"
    assert fit["execution"]["prng"] == "numpy_pcg64"
    assert fit["execution"]["fold_index_base"] == 0
    assert fit["execution"]["component_stream_keys"] == {
        "scope": "f2_random_window",
        "outer": {"purpose": "outer_test_assignment", "indices": {}},
        "inner": {"purpose": "inner_calibration_folds", "indices": {}},
    }
    assert fit["execution"]["status"] == "ready"
    assert fit["execution"]["require_trusted_file_sha256"] is True
    assert fit["execution"]["assignment_reuse"] == "load_verified_existing_or_create_once"
    assert config["protocol"] == {"status": "ready", "blocked_on": []}
    assert config["evaluation"]["per_subject_metric_role"] == "descriptive_seen_subject_only"


def test_g4_temperature_fitting_has_separate_stage_query_data_and_exact_aliases() -> None:
    config = _load("configs/experiment/generation/gen_calibration_efficiency.yaml")
    fit = config["evaluation"]["temperature_fit"]
    assert fit["stage"] == "after_adaptation_settings_freeze"
    assert fit["prediction_source"] == "all_four_strategies_on_outer_development_subject_queries"
    assert fit["support_repetitions"] == "split_manifest.repetitions.calibration"
    assert fit["query_repetitions"] == "split_manifest.repetitions.evaluation"
    assert fit["cell_fitting"] == "separate_no_smoothing_across_k"
    assert fit["inherit_f1_temperature"] == "forbidden"
    assert fit["extra_final_subject_trials"] == 0
    assert fit["final_test_fitting"] == "forbidden"
    assert fit["development_subjects_reused_after_selection"] is True
    assert fit["development_metrics_role"] == "calibration_fitting_diagnostic_only"
    assert fit["aggregate_logits_before_fitting"] == "forbidden"
    assert fit["deduplicate_aliases_before_weighting"] is True
    aliases = fit["k0_temperature_aliases"]
    assert set(aliases) == set(config["personalization"]["calibration_strategy"])
    assert aliases["real_adaptation"] == aliases["population_no_adaptation"]
    assert (
        aliases["real_plus_subject_synthetic_adaptation"]
        == aliases["real_plus_population_synthetic_adaptation"]
    )
    assert len(set(aliases.values())) == 2
    assert {
        "query_logits_sha256",
        "fold_encoder_and_generator_sha256",
        "frozen_settings_digest",
        "deployment_population_checkpoint_sha256",
        "deployment_generator_sha256",
    } <= set(fit["required_provenance"])
    assert {"macro_f1", "per_subject_macro_f1"} <= set(config["evaluation"]["metrics"])


def test_language_repeats_are_paired_and_reporting_remains_implementation_blocked() -> None:
    repeats = []
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        policy = config["training"]["repeats"]
        repeats.append(policy)
        assert policy["seed_count"] == 3
        assert policy["seed_indices"] == [0, 1, 2]
        assert policy["status"] == "ready"
        assert policy["seed_derivation"] == "temporallens_rng_v1"
        assert policy["seed_index_base"] == 0
        assert policy["rng_scope_source"] == "training.randomness.phase_scopes.reporting"
        assert policy["component_stream_keys_source"] == "training.randomness.component_stream_keys"
        assert policy["pairing"] == "seed_index"
        assert policy["within_subject_metric_reduction"] == "equal_weight_mean_over_seed_runs"
        assert policy["outcome_triggered_seed_extension"] == "forbidden"
        assert policy["logit_averaging"] == "forbidden"
        assert "training.repeats" not in config["protocol"]["blocked_on"]
        assert (
            config["evaluation"]["ece"]["temperature_source"]
            == "model_config.probability_calibration"
        )
        assert config["evaluation"]["temperature_fit"]["status"] == "ready"
        assert "evaluation.temperature_fit" not in config["protocol"]["blocked_on"]
        assert "evaluation.ece.temperature_source" not in config["protocol"]["blocked_on"]
    assert len(repeats) == 3
    assert repeats[0] == repeats[1] == repeats[2]


def test_language_rng_policy_is_shared_and_pilot_reporting_reference_one_table() -> None:
    """D70 settles declarative identities, not initialization or runner execution."""
    policies = []
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        policy = config["training"]["randomness"]
        policies.append(policy)
        assert policy == {
            "status": "ready",
            "seed_derivation": "temporallens_rng_v1",
            "derivation_function": "temporallens.utils.randomness.derive_seed",
            "root_seed_source": "experiment.seed",
            "backend": "native_pytorch",
            "seed_bridge": "unsigned_64_bit_direct_to_pytorch",
            "phase_scopes": {"pilot": "language_pilot", "reporting": "language_reporting"},
            "index_domains": {
                "seed_index": "phase_specific_zero_based_replicate",
                "development_fit_indices": list(range(8)),
                "refit_fit_index": 8,
                "epoch_index_base": 0,
                "arm_ids": {"l2": 2, "l3": 3, "l4": 4},
            },
            "component_stream_keys": {
                "data_order": ["seed_index", "fit_index", "epoch_index"],
                "head_init": ["seed_index", "fit_index"],
                "projector_init": ["seed_index", "fit_index"],
                "l3_backbone_init": ["seed_index"],
                "training_stochasticity": ["seed_index", "fit_index", "arm_id"],
                "loader_base_seed": ["seed_index", "fit_index", "epoch_index"],
                "validation_loader": ["seed_index", "fit_index", "arm_id"],
            },
            "excluded_key_fields": [
                "learning_rate",
                "candidate_index",
                "grid_size",
                "search_round",
                "horizon",
                "selected_budget",
                "config_digest",
            ],
            "initialization_sharing": "cloned_values_in_independent_modules_without_shared_storage",
            "optimizer_state": "fresh_per_fit_and_candidate",
            "construction_and_validation_consume_training_rng": "forbidden",
            "input_order": {
                "canonical_ids": "subject_then_start_sample",
                "pool": "identical_complete_eligible_window_ids_across_arms",
                "arm_specific_filtering_resampling_or_length_bucketing": "forbidden",
                "transforms": "deterministic",
            },
            "loader": {
                "sampler_and_worker_generators": "separate",
                "in_order": True,
                "persistent_workers": False,
                "training_worker_reseeding": "each_epoch_from_loader_base_seed",
                "worker_seed_bridge": "torch_initial_seed_modulo_2_pow_32_for_python_and_numpy",
                "validation_generator": "recreate_same_fit_arm_stream_each_pass",
            },
            "validation_rng": "isolate_and_restore_cpu_and_relevant_device_states",
            "revision_policy": "resolve_immutable_model_tokenizer_and_config_revisions_before_pilot",
            "required_provenance": [
                "exact_keys_derived_seeds_backend_device_and_library_versions",
                "canonical_window_pool_identity_and_per_epoch_order_digests",
                "initialization_and_frozen_backbone_state_digests",
                "actual_learning_rate_and_full_data_model_config_artifact_identities",
                "checkpoint_rng_states_and_data_order_cursor_for_resume",
            ],
        }
        for section in (
            config["training"]["learning_rate_selection"]["pilot"],
            config["training"]["repeats"],
        ):
            resolved: Any = config
            for part in section["component_stream_keys_source"].split("."):
                resolved = resolved[part]
            assert resolved == policy["component_stream_keys"]
            assert "component_stream_keys" not in section
        for gate in (
            "training.batch_execution",
            "training.learning_rate_selection",
            "training.selection",
            "evaluation.temperature_fit",
        ):
            assert gate not in config["protocol"]["blocked_on"]
        assert "training.randomness" not in config["protocol"]["blocked_on"]
        assert config["protocol"]["status"] == "blocked_pending_implementation"
    assert len(policies) == 3
    assert policies[0] == policies[1] == policies[2]


@pytest.mark.parametrize("phase", ["pilot", "reporting"])
def test_language_declared_keys_pair_only_the_approved_axes(phase: str) -> None:
    """Exercise D43 with D70 keys, without pretending a language runner exists."""
    config = _load("configs/experiment/language/adapter_llama3b_subject_split.yaml")
    policy = config["training"]["randomness"]
    scope = policy["phase_scopes"][phase]

    def seed(purpose: str, **overrides: int | float | str) -> int:
        values: dict[str, Any] = {
            "seed_index": 0,
            "fit_index": 0,
            "epoch_index": 0,
            "arm_id": 2,
        }
        values.update(overrides)
        return derive_seed(
            config["experiment"]["seed"],
            scope,
            purpose,
            {name: values[name] for name in policy["component_stream_keys"][purpose]},
        )

    for purpose in ("data_order", "head_init", "projector_init", "l3_backbone_init"):
        assert seed(purpose, arm_id=2) == seed(purpose, arm_id=3)
        assert seed(purpose, learning_rate=3e-5) == seed(purpose, learning_rate=1e-3)
        assert seed(purpose, horizon=10, selected_budget=1) == seed(
            purpose, horizon=20, selected_budget=2, config_digest="logging-only-change"
        )
    assert seed("head_init", arm_id=2) == seed("head_init", arm_id=4)
    assert seed("data_order", arm_id=2) == seed("data_order", arm_id=4)
    for purpose in ("training_stochasticity", "validation_loader"):
        assert len({seed(purpose, arm_id=arm) for arm in (2, 3, 4)}) == 3
        assert seed(purpose, learning_rate=3e-5) == seed(purpose, learning_rate=1e-3)
    for purpose in ("head_init", "projector_init", "training_stochasticity"):
        assert len({seed(purpose, fit_index=fit) for fit in (0, 4, 8)}) == 3
    assert len({seed("l3_backbone_init", fit_index=fit) for fit in (0, 4, 8)}) == 1
    assert seed("data_order", epoch_index=0) != seed("data_order", epoch_index=1)
    assert seed("loader_base_seed", epoch_index=0) != seed("loader_base_seed", epoch_index=1)
    assert seed("data_order") != seed("loader_base_seed")
    assert len({seed("head_init", seed_index=index) for index in (0, 1, 2)}) == 3
    for purpose, names in policy["component_stream_keys"].items():
        indices = {name: 0 if name != "arm_id" else 2 for name in names}
        assert derive_seed(42, "language_pilot", purpose, indices) != derive_seed(
            42, "language_reporting", purpose, indices
        )
        assert not set(names) & set(policy["excluded_key_fields"])


def test_language_approved_budget_choices_do_not_unblock_implementation() -> None:
    """D49/D50/D61 fix aggregation, exposure and peak ties, not the complete contract."""
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        selection = config["training"]["selection"]
        assert (
            selection["common_budget_rule"] == "peak_of_subject_equal_mean_cross_validation_curve"
        )
        assert selection["budget_unit"] == "epochs"
        assert selection["refit_budget_transfer"] == "same_selected_epoch_budget"
        assert selection["budget_reporting"] == {
            "epochs": "actual_training_pass_position",
            "optimizer_steps": "actual_cumulative_completed_updates",
            "batch_and_accumulation_provenance": "required",
        }
        assert selection["peak_time_tie_break"] == {
            "status": "ready",
            "comparison": "exact_unrounded_selection_curve_score",
            "tolerance": 0.0,
            "rule": "earliest_eligible_checkpoint",
        }
        scoring = config["training"]["learning_rate_selection"]["scoring"]
        source = scoring["peak_time_tie_break_source"].split(".")
        resolved = config
        for key in source:
            resolved = resolved[key]
        assert resolved == selection["peak_time_tie_break"]
        assert (
            selection["cross_validation_scheme"] == "eight_fold_subject_independent_from_manifest"
        )
        assert selection["epoch_selection"] == "per_arm_reporting_seed_peak_of_mean_curve"
        assert selection["refit_rule"] == "fresh_all32_at_same_selected_language_epoch_budget"
        assert selection["status"] == "ready"
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "training.selection" not in config["protocol"]["blocked_on"]


def test_language_approved_optimizer_and_batch_declarations_do_not_implement_them() -> None:
    """D51-D54/D72 settle optimization without implementing the agreed fitting procedure."""
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        training = config["training"]
        assert training["optimization"] == {
            "status": "ready",
            "optimizer": "adam",
            "weight_decay_mode": "coupled_l2",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "scheduler": "none",
            "parameter_groups": {
                "regularized_parameters": "trainable_projector_and_head_weight_matrices",
                "regularized_weight_decay_source": "training.weight_decay",
                "zero_decay_parameters": "trainable_bias_and_normalization_parameters",
                "zero_decay_weight_decay": 0.0,
                "normalization_exclusion": "module_type_before_tensor_shape",
                "frozen_parameters": "excluded_from_optimizer",
                "membership": "complete_and_disjoint_by_parameter_identity",
                "unclassified_trainable_parameters": "fail_before_training",
                "provenance": "parameter_names_shapes_and_group_coefficients",
            },
        }
        assert training["learning_rate"] == 0.0001
        assert training["weight_decay"] == 0.0001
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "training.optimization" not in config["protocol"]["blocked_on"]
        assert "training.batch_execution" not in config["protocol"]["blocked_on"]


def test_language_reportable_network_layouts_and_initialization_are_explicit() -> None:
    """D71 declares compact matching layouts; runtime modules are not implemented here."""
    heads = []
    projectors = []
    for path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in path:
            continue
        config = _load(path)
        head = config["head"]
        heads.append(head)
        assert head == {
            "type": "mlp",
            "num_linear_layers": 2,
            "activation": "gelu",
            "bias": True,
            "normalization": "none",
            "dropout_position": "after_hidden_activation",
            "output_activation": "none",
            "pooling": "last_token",
            "input_dim": "auto_from_llm_config",
            "hidden_dim": 256,
            "dropout": 0.1,
            "num_classes": 18,
        }
        if "adapter" in config:
            adapter = config["adapter"]
            projectors.append(adapter)
            assert adapter == {
                "type": "mlp_projector",
                "input_dim": 256,
                "num_linear_layers": 2,
                "hidden_dim": 256,
                "activation": "gelu",
                "bias": True,
                "normalization": "none",
                "dropout": 0.0,
                "output_activation": "none",
                "output_scaling": "none",
                "output_layout": "reshape_to_num_soft_tokens_by_llm_hidden_size",
                "num_soft_tokens": 8,
                "output_dim": "auto_from_llm_config",
            }
        else:
            assert "text_summary_only" in path
            assert "encoder" not in config
        backbone = config.get("llm", config.get("baseline"))
        assert backbone["freeze"] is True
        assert backbone["torch_dtype"] == "bfloat16"
        assert backbone["load_in_4bit"] is False
    assert len(heads) == 3
    assert heads[0] == heads[1] == heads[2]
    shared_spec = (REPO_ROOT / "docs/experiments/README.md").read_text()
    head_examples = re.findall(r"```yaml\n(head:\n.*?)\n```", shared_spec, re.DOTALL)
    assert len(head_examples) == 1
    assert yaml.safe_load(head_examples[0])["head"] == heads[0]
    assert len(projectors) == 2
    assert projectors[0] == projectors[1]
    projector = projectors[0]
    # A documented example width, never a replacement for auto_from_llm_config.
    example_width = 3072
    output_width = projector["num_soft_tokens"] * example_width
    hidden = projector["hidden_dim"]
    count = projector["input_dim"] * hidden + hidden + hidden * output_width + output_width
    assert count == 6_381_824


def test_language_numerical_execution_is_shared_without_unblocking_implementation() -> None:
    """D72 pins numerical declarations but does not claim a CUDA preflight or trainer."""
    policies = []
    for path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in path:
            continue
        config = _load(path)
        policy = config["training"]["numerical_execution"]
        policies.append(policy)
        assert policy == {
            "status": "ready",
            "initialization": {
                "trainable_modules": {
                    "weights_and_biases": "uniform_minus_plus_inverse_sqrt_fan_in",
                    "dtype": "float32",
                    "rng_source": "training.randomness",
                    "device_policy": "resolve_and_record_before_pilot_fixed_for_paired_runs",
                },
                "l3_backbone": {
                    "applies_to": "l3_only",
                    "construction": "pinned_transformers_from_config_never_from_pretrained",
                    "sample_dtype": "float32",
                    "final_dtype": "bfloat16",
                    "special_parameter_rules": "preserve_pinned_model_native_initialization",
                    "rng_source": "training.randomness.component_stream_keys.l3_backbone_init",
                },
                "pretrained_l2_l4_reinitialization": "forbidden",
            },
            "precision": {
                "frozen_encoder_and_preprocessing": "preserve_matching_f1_fp32_inference_contract",
                "frozen_transformer_parameters_and_floating_inputs": "bfloat16",
                "transformer_internal_promotions": "pinned_model_native_rules",
                "quantization": "none",
                "trainable_parameters_forward_gradients_and_adam_moments": "float32",
                "logits_and_classification_loss": "float32",
                "temperature_fitting_source": "evaluation.temperature_fit.execution",
                "prefix_cast_when_present": "differentiable_float32_to_bfloat16",
                "pooled_hidden_cast": "float32_before_head",
                "global_autocast": False,
                "gradient_scaling": False,
                "gradients_through_frozen_transformer": "required_when_projector_present",
            },
            "gradients": {
                "clipping": "none",
                "norm_logging": "accumulated_gradient_norm_before_update",
                "expected_trainable_gradients": "present_and_finite",
            },
            "failure": {
                "finite_before_update": [
                    "inputs",
                    "soft_prefix_when_present",
                    "pooled_features",
                    "logits",
                    "loss",
                    "accumulated_trainable_gradients",
                ],
                "finite_after_update": ["trainable_parameters", "adam_state"],
                "action": "stop_affected_phase_pending_recorded_repair",
                "pilot_candidate_failure": "stop_pilot_not_score_or_discard_candidate",
                "skipped_batches_or_updates": "forbidden",
                "dropped_or_replaced_reporting_seeds": "forbidden",
                "numerical_retries_without_recorded_repair": "forbidden",
                "silent_lr_dtype_batch_backend_or_seed_changes": "forbidden",
                "interrupted_run_resume": "exact_saved_state_only",
            },
            "reproducibility": {
                "native_bfloat16_required": True,
                "bfloat16_emulation": "forbidden",
                "deterministic_algorithms": True,
                "warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "language_float32_matmul": "ieee_float32_without_tf32",
                "frozen_f1_inference_settings": "preserve_and_verify_parent_contract",
                "attention_backend": (
                    "verify_and_pin_common_backend_before_pilot_without_validation_scores"
                ),
                "cuda_workspace_configuration": "resolve_before_cuda_context_and_record",
                "unsupported_hardware_or_operator": "block_without_silent_fallback",
                "provenance": "hardware_software_precision_backend_and_initialization_identities",
            },
        }
        assert config["training"]["batch_execution"]["status"] == "ready"
        assert config["training"]["optimization"]["status"] == "ready"
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "training.batch_execution" not in config["protocol"]["blocked_on"]
        assert "training.numerical_execution" not in config["protocol"]["blocked_on"]
        assert "training.optimization" not in config["protocol"]["blocked_on"]
        assert "training.selection" not in config["protocol"]["blocked_on"]
        source: Any = config
        for part in policy["precision"]["temperature_fitting_source"].split("."):
            source = source[part]
        assert source["dtype"] == "float64"
        assert source["status"] == "ready"
    assert len(policies) == 3
    assert policies[0] == policies[1] == policies[2]


def test_language_batch_execution_has_one_shared_actual_group_contract() -> None:
    """D73 settles batch arithmetic and weighting, not a production accumulation loop."""
    policies = []
    for path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in path:
            continue
        config = _load(path)
        training = config["training"]
        policy = training["batch_execution"]
        policies.append(policy)
        assert training["batch_size"] == 4
        assert training["gradient_accumulation_steps"] == 8
        assert policy == {
            "status": "ready",
            "microbatch_size_source": "training.batch_size",
            "accumulation_steps_source": "training.gradient_accumulation_steps",
            "drop_last": False,
            "epoch_tail": "update_using_actual_remaining_group",
            "cross_epoch_gradient_carry": "forbidden",
            "loss_normalization": {
                "objective": "class_weighted_mean_over_actual_update_group",
                "numerator": "sum_target_class_weight_times_unweighted_cross_entropy",
                "denominator": "sum_target_class_weights_in_actual_update_group",
                "denominator_timing": "precompute_from_group_training_labels_before_backward",
                "microbatch_reduction": "weighted_sum_divided_by_shared_group_denominator",
                "extra_division_by_accumulation_steps": "forbidden",
                "invalid_denominator": "fail_phase",
                "class_weights": "matching_fold_or_refit_training_only_class_weights",
                "regularization": "once_inside_adam_per_completed_update",
            },
            "update_boundaries": {
                "routine": "declared_accumulation_count_or_epoch_end",
                "gradient_reset": "after_each_completed_update",
                "optimizer_moments_across_epochs": "preserve",
                "validation_triggered_flush": "forbidden",
                "refit_budget_triggered_flush": "forbidden",
            },
            "final_horizon_endpoint": {
                "maximum": "exact_complete_maximum_epochs_without_extra_training",
                "refit_stop": (
                    "first_natural_completed_update_reaching_nominal_budget_times_own_window_count"
                ),
                "refit_partial_group_truncation": "forbidden",
                "exposure_overshoot": "strictly_less_than_batch_size_times_accumulation_steps",
            },
            "required_provenance": [
                "nominal_epoch_budget",
                "actual_forward_backward_training_window_exposure",
                "cumulative_completed_optimizer_updates",
                "actual_window_and_microbatch_counts_per_update",
                "target_class_weight_denominator_per_update",
                "epoch_tail_handling",
                "prefetched_windows_excluded_from_training_exposure",
            ],
        }
    assert len(policies) == 3
    assert policies[0] == policies[1] == policies[2]


@pytest.mark.parametrize(
    "path", [p for p in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS if "/language/" in p]
)
def test_language_decisions_are_settled_but_both_evaluation_gates_still_block(path: str) -> None:
    """Decision completion must not make an unimplemented source dispatchable."""
    config = _load(path)
    assert config["protocol"] == {
        "status": "blocked_pending_implementation",
        "implementation_status": "pending_implementation",
        "blocked_on": ["protocol.implementation_status"],
    }
    assert pending_field_paths(config) == ["protocol.implementation_status"]
    blockers = [
        "protocol.status=blocked_pending_implementation",
        "protocol.implementation_status",
    ]
    assert protocol_blockers(config) == blockers
    assert source_protocol_blockers({"source_config": str(REPO_ROOT / path)}) == [
        "source_config." + value for value in blockers
    ]


@pytest.mark.parametrize("windows", [1, 3, 4, 31, 32, 33, 35, 100, 130])
@pytest.mark.parametrize(
    "budget", [Fraction(1, 16), Fraction(1), Fraction(5, 4), Fraction(19, 2), Fraction(10)]
)
def test_d73_natural_boundaries_reach_fractional_targets_without_overrunning_horizon(
    windows: int, budget: Fraction
) -> None:
    """Check approved arithmetic on illustrative counts, not an implemented runner."""
    config = _load("configs/experiment/language/adapter_llama3b_subject_split.yaml")
    full_group = (
        config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"]
    )
    horizon = config["training"]["epochs"]
    epoch_groups = [min(full_group, windows - start) for start in range(0, windows, full_group)]
    assert sum(epoch_groups) == windows
    assert len(epoch_groups) == ceil(windows / full_group)
    if windows == 35:
        assert epoch_groups == [32, 3]
    boundaries = []
    exposure = 0
    for _ in range(horizon):
        for count in epoch_groups:
            exposure += count
            boundaries.append(exposure)
    target = budget * windows
    index = next(i for i, actual in enumerate(boundaries) if actual >= target)
    actual = boundaries[index]
    assert 0 <= actual - target < full_group
    assert actual <= ceil(budget) * windows <= horizon * windows
    complete_epochs = budget.numerator // budget.denominator
    partial_epoch = budget - complete_epochs
    assert index + 1 == (
        complete_epochs * len(epoch_groups) + ceil(partial_epoch * windows / full_group)
    )
    assert boundaries[-1] == horizon * windows
    assert len(boundaries) == horizon * ceil(windows / full_group)
    if budget == Fraction(5, 4) and windows in (100, 130):
        assert (actual, index + 1) == {100: (132, 5), 130: (194, 7)}[windows]


@pytest.mark.parametrize("group_windows", [1, 3, 5, 31, 32])
def test_d73_shared_group_denominator_matches_weighted_loss_and_parameter_gradients(
    group_windows: int,
) -> None:
    """Fixed non-signal tensors verify the objective, not batch-32 bitwise execution."""
    x = torch.arange(group_windows * 2, dtype=torch.float32).reshape(group_windows, 2)
    x = x / (group_windows * 2)
    labels = torch.arange(group_windows, dtype=torch.int64) % 3
    class_weights = torch.tensor([1.0, 3.0, 9.0])
    weight = torch.tensor([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.3]], requires_grad=True)
    bias = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
    logits = torch.nn.functional.linear(x, weight, bias)
    reference = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
    reference_gradients = torch.autograd.grad(reference, (weight, bias))
    micro_weight = weight.detach().clone().requires_grad_()
    micro_bias = bias.detach().clone().requires_grad_()
    denominator = class_weights[labels].sum()
    assert bool(torch.isfinite(denominator)) and denominator > 0
    accumulated_loss = torch.tensor(0.0)
    for start in range(0, group_windows, 4):
        end = min(group_windows, start + 4)
        micro_logits = torch.nn.functional.linear(x[start:end], micro_weight, micro_bias)
        contribution = (
            torch.nn.functional.cross_entropy(
                micro_logits, labels[start:end], weight=class_weights, reduction="sum"
            )
            / denominator
        )
        accumulated_loss += contribution.detach()
        contribution.backward()
    torch.testing.assert_close(accumulated_loss, reference.detach(), rtol=1e-5, atol=1e-6)
    for actual_gradient, expected_gradient in zip(
        (micro_weight.grad, micro_bias.grad), reference_gradients, strict=True
    ):
        assert actual_gradient is not None
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6)


def test_language_lr_grid_and_pilot_size_are_declared_without_implementing_them() -> None:
    """D73 completes the pilot declarations, not the unimplemented selection runner."""
    policies = []
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        policy = config["training"]["learning_rate_selection"]
        policies.append(policy)
        assert policy == {
            "status": "ready",
            "initial_grid": [3e-5, 1e-4, 3e-4],
            "selection_scope": "per_arm",
            "fit_scope": "development_only",
            "reporting_seed_policy": "freeze_selected_arm_learning_rate",
            "pilot": {
                "status": "ready",
                "num_folds": 2,
                "fold_policy": "shared_fixed_development_folds",
                "fold_indices": [0, 4],
                "seed_count": 1,
                "seed_index": 0,
                "rng_scope_source": "training.randomness.phase_scopes.pilot",
                "component_stream_keys_source": "training.randomness.component_stream_keys",
                "reporting_stream_policy": "separate_pilot_and_reporting_rng_streams",
                "reporting_initialization": "fresh_from_reporting_streams",
                "reporting_trajectory_reuse": "forbidden",
                "horizon_source": "training.selection.horizon",
            },
            "scoring": {
                "status": "ready",
                "metric": "macro_f1",
                "class_labels_source": "dataset.num_classes",
                "undefined_class_f1": 0.0,
                "subject_reduction": "equal_weight_mean_within_fold",
                "fold_reduction": "equal_weight_mean",
                "candidate_score": "peak_of_aggregated_curve",
                "peak_time_tie_break_source": "training.selection.peak_time_tie_break",
                "checkpoint_evaluation_source": "training.selection.checkpoint_evaluation",
            },
            "tie_break": {
                "status": "ready",
                "comparison": "exact_unrounded_candidate_score",
                "tolerance": 0.0,
                "preferred_learning_rate": 0.0001,
                "fallback": "minimum_tied_learning_rate",
            },
            "grid_edge_handling": {
                "status": "ready",
                "phase": "development_pilot_only",
                "trigger": "any_arm_selected_learning_rate_at_current_grid_endpoint",
                "winner_source": "training.learning_rate_selection.tie_break",
                "endpoint_check": "after_tie_break",
                "expansion_scope": "shared_across_l2_l3_l4",
                "maximum_expansions": 1,
                "additional_candidates": [1e-5, 1e-3],
                "expanded_grid": "sorted_union_of_initial_grid_and_additional_candidates",
                "new_candidate_initialization": "fresh_fit_per_candidate",
                "existing_pilot_results": "reuse_only_with_identical_protocol_and_artifact_identity",
                "rerank": "all_arms_over_expanded_grid",
                "repeated_endpoint": {
                    "action": "block_pending_owner_decision",
                    "block_before": [
                        "reporting_fits",
                        "all_32_refit",
                        "temperature_fitting",
                        "final_test_evaluation",
                    ],
                },
                "selected_lr_freeze": "after_shared_pilot_selection_completes",
                "reporting_or_test_trigger": "forbidden",
            },
        }
        # The existing numeric field is only a candidate, not evidence of selection.
        assert config["training"]["learning_rate"] in policy["initial_grid"]
        assert config["dataset"]["num_classes"] == 18
        assert config["experiment"]["seed"] == 42
        # D66's three reporting seeds do not change the one-seed pilot.
        assert config["training"]["repeats"]["seed_count"] == 3
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "training.learning_rate_selection" not in config["protocol"]["blocked_on"]
        assert "training.selection" not in config["protocol"]["blocked_on"]
        assert "training.repeats" not in config["protocol"]["blocked_on"]
        assert "evaluation.temperature_fit" not in config["protocol"]["blocked_on"]
    assert len(policies) == 3
    assert policies[0] == policies[1] == policies[2]
    pilot = policies[0]["pilot"]
    assert (
        len(policies) * len(policies[0]["initial_grid"]) * pilot["num_folds"] * pilot["seed_count"]
        == 18
    )


def test_language_checkpoint_evaluation_is_shared_and_keeps_execution_gated() -> None:
    """D63 declares cadence/eligibility/mapping, not a runtime selector or horizon."""
    policies = []
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        selection = config["training"]["selection"]
        policy = selection["checkpoint_evaluation"]
        policies.append(policy)
        assert policy == {
            "status": "ready",
            "interval_epochs": 0.25,
            "first_epoch_extra_checkpoints": [0.0625, 0.125],
            "smoothing": "none",
            "validation_windows": "all_eligible_query_windows_per_subject",
            "time_zero": "diagnostic_only",
            "selectable_checkpoints": "positive_time_grid_only",
            "update_boundary": {
                "mapping": "first_completed_update_reaching_nominal_training_exposure",
                "mid_accumulation_validation": "forbidden",
                "checkpoint_triggered_accumulation_flush": "forbidden",
                "duplicate_update_boundaries": "evaluate_once_record_all_nominal_targets",
            },
            "exposure_reporting": [
                "nominal_epoch_target",
                "actual_consumed_training_windows",
                "actual_epoch_position",
                "completed_optimizer_updates",
            ],
            "refit_mapping": "same_nominal_epoch_budget_on_refit_window_count",
        }
        scoring = config["training"]["learning_rate_selection"]["scoring"]
        resolved = config
        for key in scoring["checkpoint_evaluation_source"].split("."):
            resolved = resolved[key]
        assert resolved is policy
        assert "checkpoint_cadence" not in scoring
        assert "smoothing" not in scoring  # No second pilot-only copy can drift.
        assert scoring["status"] == "ready"
        assert selection["epoch_selection"] == "per_arm_reporting_seed_peak_of_mean_curve"
        assert selection["refit_rule"] == "fresh_all32_at_same_selected_language_epoch_budget"
        assert selection["status"] == "ready"
        assert config["training"]["optimization"]["status"] == "ready"
        assert config["training"]["batch_execution"]["status"] == "ready"
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "training.selection" not in config["protocol"]["blocked_on"]
    assert len(policies) == 3
    assert policies[0] == policies[1] == policies[2]


def test_language_ten_epoch_maximum_is_shared_without_unblocking_execution() -> None:
    """D64 fixes the maximum and reporting-boundary guard, not the selected B."""
    policies = []
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        training = config["training"]
        selection = training["selection"]
        policy = selection["horizon"]
        policies.append(policy)
        assert policy == {
            "status": "ready",
            "maximum_epochs_source": "training.epochs",
            "fit_scope": "learning_rate_pilot_and_reporting_selection",
            "automatic_extension": "forbidden",
            "pilot_peak_times": "diagnostic_only",
            "reporting_boundary": {
                "curve": "aggregated_reporting_selection_curve",
                "trigger": "selected_nominal_epoch_equals_maximum",
                "scope": "any_reporting_selection_curve",
                "action": "block_pending_owner_decision",
                "block_before": [
                    "all_32_refit",
                    "temperature_fitting",
                    "final_test_evaluation",
                ],
            },
        }
        pilot = training["learning_rate_selection"]["pilot"]
        resolved = config
        for key in pilot["horizon_source"].split("."):
            resolved = resolved[key]
        assert resolved is policy
        maximum = config
        for key in policy["maximum_epochs_source"].split("."):
            maximum = maximum[key]
        assert maximum == 10
        assert (
            selection["common_budget_rule"] == "peak_of_subject_equal_mean_cross_validation_curve"
        )
        assert selection["refit_budget_transfer"] == "same_selected_epoch_budget"
        assert selection["peak_time_tie_break"]["rule"] == "earliest_eligible_checkpoint"
        assert selection["epoch_selection"] == "per_arm_reporting_seed_peak_of_mean_curve"
        assert (
            selection["cross_validation_scheme"] == "eight_fold_subject_independent_from_manifest"
        )
        assert selection["refit_rule"] == "fresh_all32_at_same_selected_language_epoch_budget"
        assert training["learning_rate_selection"]["grid_edge_handling"]["status"] == "ready"
        assert training["repeats"]["seed_count"] == 3
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "training.selection" not in config["protocol"]["blocked_on"]
        assert "training.learning_rate_selection" not in config["protocol"]["blocked_on"]
    assert len(policies) == 3
    assert policies[0] == policies[1] == policies[2]


def test_language_grid_expansion_references_ties_and_has_bounded_shared_fit_count() -> None:
    """D65 declares 18 or 30 pilot fits, not a runner or extra reporting repeats."""
    language_configs = [
        _load(path) for path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS if "/language/" in path
    ]
    assert len(language_configs) == 3
    for config in language_configs:
        policy = config["training"]["learning_rate_selection"]
        edge = policy["grid_edge_handling"]
        resolved = config
        for key in edge["winner_source"].split("."):
            resolved = resolved[key]
        assert resolved is policy["tie_break"]
        assert resolved["comparison"] == "exact_unrounded_candidate_score"
        assert edge["endpoint_check"] == "after_tie_break"
        initial = policy["initial_grid"]
        additional = edge["additional_candidates"]
        assert set(initial).isdisjoint(additional)
        expanded = sorted(set(initial) | set(additional))
        assert expanded == [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        pilot = policy["pilot"]
        fits_per_candidate = len(language_configs) * pilot["num_folds"] * pilot["seed_count"]
        assert len(initial) * fits_per_candidate == 18
        assert len(additional) * fits_per_candidate == 12
        assert len(expanded) * fits_per_candidate == 30
        assert config["training"]["epochs"] == 10
        assert pilot["reporting_trajectory_reuse"] == "forbidden"
        assert policy["status"] == "ready"
        assert pilot["status"] == "ready"
        assert config["training"]["repeats"]["seed_count"] == 3
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "training.learning_rate_selection" not in config["protocol"]["blocked_on"]
        assert "training.optimization" not in config["protocol"]["blocked_on"]
        assert "training.batch_execution" not in config["protocol"]["blocked_on"]


def test_language_reporting_lineage_keeps_population_and_language_budgets_distinct() -> None:
    """D66/D67 repeat language fitting, not F1, and reuse common-B language snapshots."""
    configs = [
        _load(path) for path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS if "/language/" in path
    ]
    manifest = _load(MANIFEST_PATH)
    assert len(manifest["folds"]) == 8
    reporting_fits = 0
    for config in configs:
        training = config["training"]
        repeats = training["repeats"]
        assert repeats["seed_indices"] == list(range(repeats["seed_count"])) == [0, 1, 2]
        assert repeats["selection_scope"] == "per_arm_and_reporting_seed"
        assert repeats["population_encoder_retraining"] == "forbidden"
        assert repeats["seed_spread"] == "report_separately"
        assert repeats["inferential_unit_after_seed_reduction"] == "subject"
        reporting = training["selection"]["reporting"]
        assert reporting["status"] == "ready"
        assert reporting["scope"] == "per_arm_and_reporting_seed"
        assert reporting["split_source"] == "dataset.split_manifest"
        assert reporting["query_repetitions"] == "all_manifest_repetitions"
        assert reporting["trainable_initialization"] == "fresh_per_fold_and_refit"
        assert reporting["population_encoder_retraining"] == "forbidden"
        encoder = reporting["development_encoder"]
        assert encoder["training_budget"] == "d18_d19_accepted_refit_budget"
        assert encoder["required_device"] == "cuda"
        expected_consumption = (
            "normalization_and_class_weights_only"
            if "encoder" not in config
            else "encoder_normalization_and_class_weights"
        )
        assert encoder["consumption"] == expected_consumption
        reuse = reporting["development_checkpoint_reuse"]
        assert (
            reuse["source"] == "accepted_language_selection_trajectory_at_selected_language_budget"
        )
        assert reuse["snapshot_contents"] == "all_trainable_parameters_and_model_buffers"
        assert reuse["frozen_parent_identity"] == "required"
        assert reuse["training_prefix_horizon_and_validation_independence"] == "required"
        assert reuse["validation_changes_training_rng_or_model_state"] == "forbidden"
        assert reuse["missing_or_incompatible_snapshot"] == "block_artifact_creation"
        assert reporting["refit"]["budget_source"] == "training.selection.epoch_selection"
        assert reporting["refit"]["trainable_and_optimizer_state"] == "fresh"
        assert reporting["refit"]["required_population_device"] == "cuda"
        assert reporting["pre_refit_and_temperature_guard"] == (
            "all_reporting_curves_pass_horizon_boundary_check"
        )
        reporting_fits += repeats["seed_count"] * (len(manifest["folds"]) + 1)
    assert reporting_fits == 81
    assert reporting_fits + 18 == 99
    assert reporting_fits + 30 == 111


def test_l3_reporting_backbone_is_fixed_within_seed_not_across_seeds_or_from_pilot() -> None:
    """D68 matches a seed's fold/refit feature extractor without changing pretrained arms."""
    l3 = _load("configs/experiment/language/adapter_random_transformer.yaml")
    policy = l3["baseline"]["reporting_backbone"]
    assert policy == {
        "status": "ready",
        "scope": "per_reporting_seed",
        "sharing": "same_immutable_state_across_development_folds_and_refit",
        "architecture_and_dtype": "match_l2_frozen_backbone",
        "required_identity": [
            "architecture_revision",
            "dtype",
            "model_state_sha256",
            "reporting_seed_index",
        ],
        "pilot_artifact_reuse": "forbidden",
        "initialization_source": "training.numerical_execution.initialization.l3_backbone",
    }
    for path in (
        "configs/experiment/language/adapter_llama3b_subject_split.yaml",
        "configs/experiment/language/adapter_text_summary_only.yaml",
    ):
        assert "reporting_backbone" not in _load(path).get("baseline", {})


def test_language_refit_temperatures_and_cross_fitted_diagnostics_have_separate_lineages() -> None:
    """D69 never scores pooled T on its fitting predictions as held-out validation."""
    for path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in path:
            continue
        config = _load(path)
        fit = config["evaluation"]["temperature_fit"]
        assert fit["scope"] == "per_arm_and_reporting_seed"
        assert fit["prediction_source"] == (
            "pooled_subject_out_of_fold_language_models_at_selected_budget_per_arm_seed"
        )
        assert (
            fit["checkpoint_source"] == "training.selection.reporting.development_checkpoint_reuse"
        )
        assert fit["matching_refit_source"] == "training.selection.reporting.refit"
        assert fit["query_repetitions"] == "all_manifest_repetitions"
        for key in (
            "across_seed_pooling",
            "temperature_averaging",
            "scoring_own_fitting_predictions_as_validation",
        ):
            assert fit[key] == "forbidden"
        assert fit["artifact"] == "model_config.probability_calibration"
        assert fit["status"] == "ready"
        assert fit["execution"]["status"] == "ready"
        execution = config["evaluation"]["calibration_execution"]
        assert execution["validation_temperature_source"] == (
            "leave_one_subject_out_within_common_budget_outer_fold"
        )
        diagnostic = execution["validation_temperature_fit"]
        assert (
            diagnostic["model_checkpoint"] == "same_fold_arm_seed_common_language_budget_checkpoint"
        )
        assert diagnostic["objective_and_weighting_source"] == "evaluation.temperature_fit"
        assert diagnostic["numerical_execution_source"] == "evaluation.temperature_fit.execution"
        assert diagnostic["stage"] == "after_all_reporting_budgets_freeze_and_boundary_check"
        assert diagnostic["fit_subjects"] == "other_three_outer_validation_subjects"
        assert diagnostic["score_subject"] == "excluded_outer_validation_subject"
        assert diagnostic["fits_per_fold"] == 4
        assert diagnostic["fold_summary_reduction"] == "equal_weight_mean"
        assert diagnostic["retain_subject_seed_metrics"] is True
        assert (
            diagnostic["inference_reduction_source"]
            == "training.repeats.within_subject_metric_reduction"
        )
        assert diagnostic["rotation"] == "each_of_four_outer_validation_subjects"
        assert diagnostic["model_retraining"] == "forbidden"
        assert diagnostic["reuse_for_refit_temperature"] == "forbidden"
        assert diagnostic["interpretation"] == "conditional_development_diagnostic_not_nested_cv"
        assert diagnostic["persist_model_fitting_scoring_subjects_and_temperature"] is True
        assert config["protocol"]["status"] == "blocked_pending_implementation"
        assert "evaluation.temperature_fit" not in config["protocol"]["blocked_on"]


def test_language_pilot_folds_resolve_only_development_subjects() -> None:
    """D57 resolves fold IDs through D16, never a second subject split or final-test pool."""
    manifest = _load(MANIFEST_PATH)
    folds = {fold["index"]: fold["validation_subjects"] for fold in manifest["folds"]}
    development = set(manifest["training_subjects"])
    held_out = set(manifest["test_subjects"])
    for relative_path in REPORTABLE_SUBJECT_INDEPENDENT_CONFIGS:
        if "/language/" not in relative_path:
            continue
        config = _load(relative_path)
        assert config["dataset"]["split_manifest"] == MANIFEST_PATH
        pilot = config["training"]["learning_rate_selection"]["pilot"]
        indices = pilot["fold_indices"]
        assert len(indices) == len(set(indices)) == pilot["num_folds"] == 2
        queries = [folds[index] for index in indices]
        assert queries == [[1, 11, 21, 31], [6, 16, 26, 36]]
        query_subjects = {subject for subjects in queries for subject in subjects}
        assert len(query_subjects) == 8
        assert query_subjects <= development
        assert query_subjects.isdisjoint(held_out)
        for subjects in queries:
            training_subjects = development - set(subjects)
            assert len(training_subjects) == 28
            assert training_subjects.isdisjoint(held_out)


def test_population_temperature_fits_use_same_objective_and_weighting() -> None:
    for relative_path in ECE_CONFIGS:
        config = _load(relative_path)
        fit = config["evaluation"].get("temperature_fit")
        if fit is None:  # robustness consumes an existing fit
            continue
        assert fit["objective"] == "ordinary_negative_log_likelihood"
        assert fit["class_weighting"] == "none"
        assert fit["subject_weighting"] == "equal"
        assert fit["repeat_weighting"] == "equal_within_subject"
        assert fit["window_weighting"] == "equal_within_repeat"
        assert fit["freeze_before_final_test"] is True
        assert (
            fit["execution"]["invalid_or_unresolved_boundary_fit"] == "block_reportable_evaluation"
        )


@pytest.mark.parametrize(
    "relative_path", [path for path in ECE_CONFIGS if "/robustness_" not in path]
)
def test_shared_temperature_numerics_are_fixed_without_resolving_arm_designs(
    relative_path: str,
) -> None:
    config = _load(relative_path)
    execution = config["evaluation"]["temperature_fit"]["execution"]
    expected = {
        "solver": "scipy_minimize_scalar_bounded",
        "parameter": "inverse_temperature",
        "dtype": "float64",
        "bounds": [0.01, 100.0],
        "tolerance": 1e-8,
        "max_iterations": 500,
        "nll_evaluation": "stable_logsumexp",
        "fitting_weights": "normalized_d33_weights",
        "boundary_check": "weighted_nll_derivative_at_bounds",
        "nll_non_regression_reference": "temperature_1",
        "nll_non_regression_tolerance": 1e-8,
        "persist_solver_diagnostics": True,
        "invalid_or_unresolved_boundary_fit": "block_reportable_evaluation",
    }
    assert {key: execution[key] for key in expected} == expected
    if "/baseline_cnn_random_split" not in relative_path:
        assert execution["status"] == "ready"
        assert "evaluation.temperature_fit.execution" not in config["protocol"]["blocked_on"]


def test_g0_subject_vectors_exercise_a_controlled_structural_check() -> None:
    config = _load("configs/experiment/generation/gen_vae_debug.yaml")
    diagnostic = config["evaluation"]["subject_conditioning_diagnostic"]
    assert config["generator"]["subject_conditioning"] == "population_only"
    # G0 exists to exercise G1's path, so it mirrors G1's key layout and dimension.
    g1 = _load("configs/experiment/generation/gen_vae_train.yaml")["generator"]
    assert config["generator"]["subject_embedding"]["output_dim"] == 16
    assert config["generator"]["subject_embedding"]["output_dim"] == (
        g1["subject_embedding"]["output_dim"]
    )
    assert diagnostic["dimension_source"] == "generator.subject_embedding.output_dim"
    assert diagnostic["vectors"] == ["zero_vector", "first_unit_basis_vector"]
    assert diagnostic["hold_fixed"] == ["class_label", "latent_noise_draw", "decoder_eval_mode"]
    assert diagnostic["result_role"] == "structural_sensitivity_only"
    assert "subject_embedding_changes_sample" in config["evaluation"]["checks"]


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
    for target in g3_targets:
        assert target["temperature_source"] == "frozen_strategy_k_temperature_table"
        assert target["calibration_source_config"] == (
            "configs/experiment/generation/gen_calibration_efficiency.yaml"
        )
        assert (
            target["temperature_run_selection"] == "pending_matching_immutable_calibration_run_id"
        )
        assert target["temperature_lookup_axes"] == ["strategy", "calibration_trials"]
        assert target["require_matching_model_generator_and_settings_digests"] is True
    assert all(
        target["temperature_source"] == "model_config.probability_calibration"
        for target in registry["targets"]
        if target["arm"] != "generation"
    )
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


#: Keys that declare which budget built a development artifact, by the kind of artifact named.
_ARTIFACT_BUDGET_KEYS = {
    "encoder": ("training_budget", "model_training_budget", "encoder_training_budget"),
    "generator": ("generator_training_budget",),
}


def _fixed_budget_paths(node: Any) -> list[tuple[str, str]]:
    """Every ``(key, path)`` below ``node`` whose value names a fixed-budget development artifact."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and value.endswith("/fixed_budget.pt"):
                found.append((str(key), value))
            else:
                found.extend(_fixed_budget_paths(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_fixed_budget_paths(value))
    return found


def _budget_declarations(node: Any, out: dict[str, set[str]]) -> None:
    """Record the budget each mapping declares for the artifacts named anywhere beneath it."""
    if isinstance(node, dict):
        for key, path in _fixed_budget_paths(node):
            kind = "generator" if "generator" in key else "encoder"
            for budget_key in _ARTIFACT_BUDGET_KEYS[kind]:
                if isinstance(node.get(budget_key), str):
                    out.setdefault(path, set()).add(node[budget_key])
        for value in node.values():
            _budget_declarations(value, out)
    elif isinstance(node, list):
        for value in node:
            _budget_declarations(value, out)


def test_every_config_naming_a_shared_artifact_agrees_on_its_budget() -> None:
    """One path, one model.

    Several experiments name the same development artifact (D35): the complement-28 encoder and
    generator at ``development/fold{k}/fixed_budget.pt``. The path identifies a file, not what
    built it, so a config naming that path with a different budget describes a different model —
    and whichever experiment ran second would silently load the first one's. This once held two
    undefined budget names for the same encoder. Every config naming such a path must declare its
    budget, and all of them must declare the same one.
    """
    by_path: dict[str, dict[str, set[str]]] = {}
    for config_path in sorted(REPO_ROOT.glob("configs/experiment/**/*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        relative = str(config_path.relative_to(REPO_ROOT))
        named = {path for _, path in _fixed_budget_paths(config)}
        declared: dict[str, set[str]] = {}
        _budget_declarations(config, declared)
        for path in named:
            assert declared.get(path), f"{relative} names {path} but declares no budget for it"
            by_path.setdefault(path, {})[relative] = declared[path]

    assert by_path, "no fixed-budget development artifacts found; the walk is broken"
    for path, per_config in by_path.items():
        budgets = set().union(*per_config.values())
        assert len(budgets) == 1, f"{path} is built with different budgets: {per_config}"


def test_shared_f1_development_path_has_one_encoder_checkpoint_source() -> None:
    """A shared path must not mean a retained trajectory to one consumer and a fresh fit to another."""
    f1 = _load("configs/experiment/foundation/baseline_cnn_subject_split.yaml")
    expected = f1["training"]["development_checkpoints"]["source"]
    path = f1["evaluation"]["temperature_fit"]["checkpoint_pattern"]
    consumers: list[tuple[str, str]] = []

    def inspect(node: Any, config_path: str) -> None:
        if isinstance(node, dict):
            if path in node.values():
                assert "encoder_checkpoint_source" in node, config_path
                consumers.append((config_path, node["encoder_checkpoint_source"]))
            for value in node.values():
                inspect(value, config_path)
        elif isinstance(node, list):
            for value in node:
                inspect(value, config_path)

    for config_path in sorted(REPO_ROOT.glob("configs/experiment/**/*.yaml")):
        inspect(yaml.safe_load(config_path.read_text()), str(config_path.relative_to(REPO_ROOT)))
    assert len(consumers) == 7, consumers  # F1, G2-G4 and L2-L4 share this exact artifact.
    assert all(source == expected for _, source in consumers), consumers


def test_f2_draws_its_windows_from_the_development_subjects_only() -> None:
    """D41: F2 is a population fit, so the held-out subjects stay out of it.

    The leakage F2 demonstrates comes from splitting windows within subjects; it needs no held-out
    subject to show, and nothing about the demonstration is weakened by excluding them.
    """
    dataset = _load("configs/experiment/foundation/baseline_cnn_random_split.yaml")["dataset"]
    assert dataset["split"] == "random_window"
    assert dataset["split_manifest"] == MANIFEST_PATH
    assert dataset["subject_pool"] == "split_manifest.training_subjects"


def test_no_explicit_subject_subset_contains_a_held_out_subject() -> None:
    """D41's invariant where a config names subjects directly rather than through the manifest.

    Debug subsets are written out by hand, which is exactly where a held-out subject could slip
    into a training fit unnoticed. The manifest's own structural checks cover every config that
    goes through it; this covers the ones that do not.
    """
    from temporallens.data.splits import SplitManifest

    held_out = set(SplitManifest.load(REPO_ROOT / MANIFEST_PATH).test_subjects)
    offenders = []
    for config_path in sorted(REPO_ROOT.glob("configs/experiment/**/*.yaml")):
        debug_split = (yaml.safe_load(config_path.read_text()).get("dataset") or {}).get(
            "debug_split"
        )
        if not debug_split:
            continue
        named = set(debug_split.get("train_subjects", [])) | set(
            debug_split.get("held_out_subjects", [])
        )
        if named & held_out:
            offenders.append(f"{config_path.relative_to(REPO_ROOT)}: {sorted(named & held_out)}")
    assert not offenders, f"debug subsets use final held-out subjects: {offenders}"

"""The F1 development harness, end to end on DB2-shaped doubles (never real or test-subject data).

The full run trains real models, but a scripted trainer wrapper replaces each fold's validation
curve so the D18 selection and the common budget are known in advance: fold peaks
12, 14, 15, 15, 16, 18, 20 and 25 give B = ceil((15 + 16) / 2) = 16. Everything else — snapshots,
artifacts, temperatures, the refit — runs for real. The F1 artifact verifier is tested here too,
as the consumer of those same artifacts, so producer/consumer agreement costs no second run.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from _synthetic import write_db2_like_development_set

from temporallens.data.splits import SplitManifest
from temporallens.models.checkpoint import (
    checkpoint_sha256,
    load_checkpoint,
    rebuild_model,
    refit_checkpoint_path,
)
from temporallens.models.encoders.cnn1d import build_model
from temporallens.preprocessing.normalize import ChannelStats
from temporallens.training import f1_artifacts as fa
from temporallens.training import subject_cv as cv
from temporallens.training.class_weights import class_weights_sha256
from temporallens.training.selection import ESCALATED_HORIZON_EPOCHS, INITIAL_HORIZON_EPOCHS

MANIFEST = SplitManifest.load()
PEAKS = (12, 14, 15, 15, 16, 18, 20, 25)
BUDGET = 16
REPO_ROOT = Path(__file__).resolve().parents[1]
F1_CONFIG = REPO_ROOT / "configs/experiment/foundation/baseline_cnn_subject_split.yaml"


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _config(data_dir: Path, **overrides: Any) -> cv.SubjectCVConfig:
    values: dict[str, Any] = {
        "name": "f1_test",
        "seed": 42,
        "data_dir": data_dir,
        "manifest": MANIFEST,
        "window_size": 100,
        "stride": 50,
        "hidden_dim": 4,
        "embedding_dim": 8,
        "dropout": 0.2,
        "batch_size": 64,
        "learning_rate": 3e-3,
        "weight_decay": 1e-4,
        "device": "cpu",
    }
    values.update(overrides)
    return cv.SubjectCVConfig(**values)


def _peaked(peak: int, epochs: int) -> list[float]:
    """A curve whose only full-window maximum ends exactly at ``peak``."""
    return [0.9 if peak - 9 <= epoch <= peak else 0.1 for epoch in range(1, epochs + 1)]


def _scripted_trainer(peaks: tuple[int, ...]) -> cv.Trainer:
    def trainer(
        config: cv.SubjectCVConfig, pool: cv.DevelopmentPool, **kwargs: Any
    ) -> cv.FitResult:
        fit = cv.fit_population_model(config, pool, **kwargs)
        validation = tuple(kwargs["validation_subjects"])
        if validation:
            fold = MANIFEST.fold_validation_subjects.index(validation)
            fit.raw_scores = _peaked(peaks[fold], kwargs["epochs"])
        return fit

    return trainer


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """Tiny CPU convolutions run about twice as fast on one thread as on four."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Short 100-sample windows keep the full eight-fold run fast; these are never reportable.
    return write_db2_like_development_set(
        tmp_path_factory.mktemp("db2_like"), samples_per_segment=150, terminal_rest=150
    )


@pytest.fixture(scope="module")
def pool(data_dir: Path) -> cv.DevelopmentPool:
    return cv.load_development_pool(_config(data_dir))


@pytest.fixture(scope="module")
def run(data_dir: Path, pool: cv.DevelopmentPool, tmp_path_factory: pytest.TempPathFactory):
    base = tmp_path_factory.mktemp("f1_run")
    result = cv.run_subject_cross_validation(
        _config(data_dir),
        checkpoint_base=base / "checkpoints",
        output_dir=base / "runs",
        trainer=_scripted_trainer(PEAKS),
        pool=pool,
    )
    return result, base


def test_the_run_selects_each_fold_and_the_common_budget(run) -> None:
    result, _ = run
    assert [s.epoch for s in result.selections] == list(PEAKS)
    assert result.budget == BUDGET
    assert result.accepted_horizon == INITIAL_HORIZON_EPOCHS
    assert [d.action for d in result.decisions] == ["accept"]
    assert result.summary["refit_epochs"] == BUDGET
    assert result.summary["final_test_evaluated"] is False


def test_a_preliminary_run_never_writes_canonical_paths(run) -> None:
    """D47: only a reportable run writes where downstream consumers read."""
    result, base = run
    assert result.lineage == cv.PRELIMINARY
    assert result.summary["lineage"] == cv.PRELIMINARY
    for path in result.artifacts.values():
        assert path.is_relative_to(base / "checkpoints" / "preliminary")
    assert not (base / "checkpoints" / "f1_test").exists()


def test_fixed_budget_models_are_each_trajectory_at_b_not_its_own_best(run) -> None:
    """D48: development artifacts read every fold at the shared B, with full provenance."""
    result, _ = run
    for k in range(MANIFEST.num_folds):
        _, best = load_checkpoint(result.artifacts[f"fold{k}_best"])
        state, fixed = load_checkpoint(result.artifacts[f"fold{k}_fixed_budget"])
        provenance = fixed["provenance"]
        assert provenance["role"] == "common_budget_development"
        assert provenance["source_epoch"] == provenance["training_budget"] == BUDGET
        assert provenance["accepted_horizon"] == INITIAL_HORIZON_EPOCHS
        # best.pt keeps its own epoch and records the campaign's B separately, never relabelled.
        assert best["provenance"]["source_epoch"] == PEAKS[k]
        assert best["provenance"]["common_budget"] == provenance["common_budget"] == BUDGET
        # Fold 4 peaks at B itself: the two roles then share one snapshot but stay distinct files.
        same_epoch = best["provenance"]["source_checkpoint_sha256"] == (
            provenance["source_checkpoint_sha256"]
        )
        assert same_epoch == (PEAKS[k] == BUDGET)
        assert best["provenance"]["role"] == "selected_fold_analysis"
        assert provenance["scoring_subject_ids"] == list(MANIFEST.validation_subjects(k))
        assert provenance["training_subject_ids"] == list(MANIFEST.training_subjects_for_fold(k))
        assert provenance["manifest_hash"] == MANIFEST.manifest_hash
        assert provenance["device"] == "cpu"
        assert len(provenance["class_weights_sha256"]) == 64
        # BatchNorm buffers travel with the parameters.
        assert any(name.endswith("running_mean") for name in state)
        assert "probability_calibration" not in fixed  # its T lives in the refit, not here


def test_fold_statistics_come_from_that_folds_training_subjects_only(run) -> None:
    """§3.3: normalization and class weights are refitted on each fold's 28 subjects."""
    result, _ = run
    for k in range(MANIFEST.num_folds):
        _, config = load_checkpoint(result.artifacts[f"fold{k}_fixed_budget"])
        expected = f"train_subjects={sorted(MANIFEST.training_subjects_for_fold(k))}"
        assert config["normalization"]["fitted_on"] == expected
    _, refit = load_checkpoint(result.artifacts["refit"])
    assert (
        refit["normalization"]["fitted_on"]
        == f"train_subjects={sorted(MANIFEST.training_subjects)}"
    )


def test_no_artifact_names_a_final_test_subject(run) -> None:
    """The harness only ever loads development subjects; the test eight are absent on disk."""
    result, _ = run
    test_subjects = set(MANIFEST.test_subjects)
    for path in result.artifacts.values():
        _, config = load_checkpoint(path)
        named = set(config["provenance"].get("training_subject_ids", []))
        named |= set(config["provenance"].get("scoring_subject_ids", []))
        named |= set(config.get("probability_calibration", {}).get("subject_ids", []))
        assert not named & test_subjects


def test_the_refit_carries_the_pooled_out_of_fold_temperature(run) -> None:
    """D33/D39/D81: one T from the eight fixed-budget models' OOF logits, repetitions 1–6."""
    result, _ = run
    _, refit = load_checkpoint(result.artifacts["refit"])
    calibration = refit["probability_calibration"]
    assert calibration["status"] == "fitted"
    assert calibration["prediction_source"] == "pooled_subject_out_of_fold_fixed_budget_models"
    assert calibration["subject_ids"] == sorted(MANIFEST.training_subjects)
    assert calibration["repetition_ids"] == [1, 2]
    assert calibration["excluded_repetitions"] == [0]
    assert calibration["fold_artifact_sha256"] == [
        checkpoint_sha256(result.artifacts[f"fold{k}_fixed_budget"])
        for k in range(MANIFEST.num_folds)
    ]
    assert calibration["temperature"] == pytest.approx(result.refit_temperature.temperature)
    assert refit["provenance"]["training_budget"] == BUDGET
    assert refit["provenance"]["training_subject_ids"] == list(MANIFEST.training_subjects)
    assert isinstance(rebuild_model(result.artifacts["refit"]), torch.nn.Module)


def test_selected_fold_models_have_their_own_temperature_and_d40_diagnostics(run) -> None:
    result, _ = run
    for k, record in enumerate(result.summary["folds"]):
        _, best = load_checkpoint(result.artifacts[f"fold{k}_best"])
        calibration = best["probability_calibration"]
        assert calibration["role"] == "selected_fold_extended_analysis_only"
        assert calibration["subject_ids"] == sorted(MANIFEST.validation_subjects(k))
        diagnostics = record["cross_fitted_validation_diagnostics"]
        assert diagnostics["status"] == "fitted"
        # README §D40: the model identity, so every fit is tied to one frozen best.pt.
        identity = diagnostics["model_identity"]
        assert identity["checkpoint_sha256"] == checkpoint_sha256(result.artifacts[f"fold{k}_best"])
        assert identity["checkpoint_sha256"] == record["selected_checkpoint_sha256"]
        assert identity["source_epoch"] == PEAKS[k] and identity["common_budget"] == BUDGET
        for scored, entry in diagnostics["per_subject"].items():
            assert entry["scored_subject"] == int(scored)
            assert int(scored) not in entry["fitting_subjects"]
            assert len(entry["fitting_subjects"]) == 3
            # The persisted weights: each fitting subject 1/3, its two repetitions equal, rep 0 off.
            totals = entry["fitting_weight_by_subject_repetition"]
            assert sorted(int(s) for s in totals) == entry["fitting_subjects"]
            for cells in totals.values():
                assert cells["0"] == 0.0
                assert cells["1"] == pytest.approx(1 / 6) and cells["2"] == pytest.approx(1 / 6)
            assert len(entry["fitting_weights_sha256"]) == 64
            assert entry["fitting_logits_sha256"] != entry["fitting_weights_sha256"]


def test_a_second_run_refuses_to_replace_frozen_artifacts(run, data_dir, pool) -> None:
    _, base = run
    with pytest.raises(cv.ArtifactExistsError):
        cv.run_subject_cross_validation(
            _config(data_dir),
            checkpoint_base=base / "checkpoints",
            output_dir=base / "runs",
            pool=pool,
        )


def test_validation_does_not_change_the_training_trajectory(data_dir, pool) -> None:
    """D48's premise: a fit validated every epoch ends bitwise where an unvalidated one does."""
    config = _config(data_dir)
    fold = MANIFEST.training_subjects_for_fold(0)
    validated = cv.fit_population_model(
        config,
        pool,
        train_subjects=fold,
        validation_subjects=MANIFEST.validation_subjects(0),
        epochs=3,
        device=torch.device("cpu"),
    )
    silent = cv.fit_population_model(
        config, pool, train_subjects=fold, epochs=3, device=torch.device("cpu")
    )
    for name, tensor in validated.model.state_dict().items():
        assert torch.equal(tensor, silent.model.state_dict()[name]), name


def test_the_training_prefix_does_not_depend_on_the_horizon(data_dir, pool, tmp_path) -> None:
    """Epoch 10 of an 11-epoch fit equals epoch 10 of a 12-epoch fit."""
    config = _config(data_dir)
    kwargs = {
        "train_subjects": MANIFEST.training_subjects_for_fold(1),
        "device": torch.device("cpu"),
    }
    short = cv.fit_population_model(config, pool, epochs=11, snapshot_dir=tmp_path / "a", **kwargs)
    long = cv.fit_population_model(config, pool, epochs=12, snapshot_dir=tmp_path / "b", **kwargs)
    a = cv.load_snapshot(config, short, 10).state_dict()
    b = cv.load_snapshot(config, long, 10).state_dict()
    assert all(torch.equal(a[name], b[name]) for name in a)


def test_a_missing_or_changed_snapshot_is_refused(data_dir, pool, tmp_path) -> None:
    config = _config(data_dir)
    fit = cv.fit_population_model(
        config,
        pool,
        train_subjects=MANIFEST.training_subjects_for_fold(2),
        epochs=10,
        snapshot_dir=tmp_path,
        device=torch.device("cpu"),
    )
    with pytest.raises(cv.SnapshotError, match="no retained snapshot"):
        cv.load_snapshot(config, fit, 9)  # ineligible epochs are never retained
    path, _ = fit.snapshots[10]
    torch.save({}, path)
    with pytest.raises(cv.SnapshotError, match="changed"):
        cv.load_snapshot(config, fit, 10)


def test_a_non_finite_loss_fails_the_fit_before_any_update(data_dir, pool, monkeypatch) -> None:
    """D38: a NaN loss raises before the optimizer steps; the update is never silently skipped."""
    config = _config(data_dir)

    def poisoned(spec: dict[str, Any]) -> torch.nn.Module:
        model = build_model(spec)
        with torch.no_grad():
            model.head[1].bias.fill_(float("nan"))
        return model

    def never_step(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the optimizer stepped on a non-finite loss")

    monkeypatch.setattr(cv, "build_model", poisoned)
    monkeypatch.setattr(torch.optim.Adam, "step", never_step)
    with pytest.raises(cv.NonFiniteTrainingError, match="no update made"):
        cv.fit_population_model(
            config,
            pool,
            train_subjects=MANIFEST.training_subjects_for_fold(0),
            epochs=1,
            device=torch.device("cpu"),
        )


def _fake_fit(epochs: int, peak: int) -> cv.FitResult:
    return cv.FitResult(
        model=build_model(_config(Path(".")).model_spec()),
        normalization=ChannelStats(
            mean=np.zeros(12, np.float32),
            std=np.ones(12, np.float32),
            fitted_on="fake",
            num_samples=1,
        ),
        class_weights=np.ones(18, np.float32),
        epochs=epochs,
        epoch_losses=[0.0] * epochs,
        raw_scores=_peaked(peak, epochs),
        per_subject_scores=[],
        snapshots={},
        train_window_count=1,
        window_identity_sha256="0" * 64,
        rng_provenance={},
    )


def test_d19_escalates_once_then_blocks_without_a_refit(pool, tmp_path) -> None:
    """A budget at the horizon reruns all eight folds at 60; a second hit stops, no refit."""
    calls: list[tuple[int, bool]] = []

    def boundary(
        config: cv.SubjectCVConfig, _pool: cv.DevelopmentPool, **kwargs: Any
    ) -> cv.FitResult:
        calls.append((kwargs["epochs"], bool(kwargs["validation_subjects"])))
        return _fake_fit(kwargs["epochs"], peak=kwargs["epochs"])

    with pytest.raises(cv.HorizonBlockedError):
        cv.run_subject_cross_validation(
            _config(tmp_path),
            checkpoint_base=tmp_path / "checkpoints",
            output_dir=tmp_path / "runs",
            trainer=boundary,
            pool=pool,
        )
    assert calls == [(INITIAL_HORIZON_EPOCHS, True)] * 8 + [(ESCALATED_HORIZON_EPOCHS, True)] * 8
    assert not (tmp_path / "checkpoints").exists()


def test_scope_not_device_or_clean_tree_decides_the_lineage(tmp_path) -> None:
    """A preliminary run is preliminary on CUDA too, even from a clean tree."""
    cuda = torch.device("cuda")
    for clean in (True, False):
        assert cv.execution_lineage(cv.PRELIMINARY, cuda, clean_tree_required=clean) == (
            cv.PRELIMINARY
        )
    with pytest.raises(ValueError, match="unknown execution scope"):
        cv.execution_lineage("canonical", cuda, clean_tree_required=True)
    assert cv.lineage_root(tmp_path, cv.CANONICAL) == tmp_path
    assert cv.lineage_root(tmp_path, cv.PRELIMINARY) == tmp_path / "preliminary"


def test_a_reportable_run_is_refused_until_the_final_test_stage_exists(pool, tmp_path) -> None:
    """Handoff §6: no canonical artifact before the final-test stage is written and rehearsed."""
    assert cv.REPORTABLE_EXECUTION_BLOCKERS
    with pytest.raises(cv.ReportableExecutionError, match="final-test stage"):
        cv.execution_lineage(cv.REPORTABLE, torch.device("cuda"), clean_tree_required=True)

    def never(*_: Any, **__: Any) -> cv.FitResult:
        raise AssertionError("a refused reportable run must not train")

    with pytest.raises(cv.ReportableExecutionError):
        cv.run_subject_cross_validation(
            _config(tmp_path),
            checkpoint_base=tmp_path / "checkpoints",
            output_dir=tmp_path / "runs",
            scope=cv.REPORTABLE,
            require_clean_tree=True,
            trainer=never,
            pool=pool,
        )
    assert not (tmp_path / "checkpoints").exists() and not (tmp_path / "runs").exists()


def test_once_unblocked_a_reportable_run_is_still_cuda_and_clean_tree_only(monkeypatch) -> None:
    """The remaining guards hold for the stage that eventually lifts the blockers."""
    monkeypatch.setattr(cv, "REPORTABLE_EXECUTION_BLOCKERS", ())
    cuda = torch.device("cuda")
    assert cv.execution_lineage(cv.REPORTABLE, cuda, clean_tree_required=True) == cv.CANONICAL
    with pytest.raises(cv.ReportableExecutionError, match="CUDA-only"):
        cv.execution_lineage(cv.REPORTABLE, torch.device("mps"), clean_tree_required=True)
    with pytest.raises(cv.ReportableExecutionError, match="clean working tree"):
        cv.execution_lineage(cv.REPORTABLE, cuda, clean_tree_required=False)


def test_artifacts_carry_every_provenance_key_the_f1_config_requires(run) -> None:
    """D48 and D33 list what must be recorded; the harness must write each one."""
    f1 = _yaml(F1_CONFIG)
    source_keys = f1["training"]["development_checkpoints"]["required_source_provenance"]
    calibration_keys = f1["evaluation"]["temperature_fit"]["required_provenance"]
    result, _ = run
    for k in range(MANIFEST.num_folds):
        _, fixed = load_checkpoint(result.artifacts[f"fold{k}_fixed_budget"])
        assert set(source_keys) <= set(fixed["provenance"]), k
    _, refit = load_checkpoint(result.artifacts["refit"])
    assert set(calibration_keys) <= set(refit["probability_calibration"])
    assert refit["provenance"]["class_weights_sha256"]  # D85: the refit parent carries it too


# --- consumption: the shared F1 artifact verifier ---------------------------------------------


def _expect(role: str, *, fold: int | None = None, **overrides: Any) -> fa.F1Expectation:
    """A consumer's expectation of the scripted run, before any deliberate mistake."""
    values: dict[str, Any] = {
        "role": role,
        "budget": BUDGET,
        "horizon": INITIAL_HORIZON_EPOCHS,
        "lineage": cv.PRELIMINARY,
        "manifest": MANIFEST,
        "fold": fold,
    }
    if role == cv.SELECTED_FOLD_ANALYSIS and fold is not None:
        values["selected_epoch"] = PEAKS[fold]
    values.update(overrides)
    return fa.F1Expectation(**values)


def _rewrite(source: Path, target: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """A copy of ``source`` with its ``model_config`` edited: a tampered or mislabelled artifact."""
    payload = torch.load(source, map_location="cpu", weights_only=True)
    edit(payload["model_config"])
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def test_the_verifier_accepts_every_artifact_the_harness_wrote(run, pool) -> None:
    """Producer/consumer agreement over the whole run, with B recomputed from the eight epochs."""
    result, base = run
    verified = fa.verify_f1_run(
        base / "checkpoints",
        "f1_test",
        selected_epochs=PEAKS,
        horizon=INITIAL_HORIZON_EPOCHS,
        lineage=cv.PRELIMINARY,
        manifest=MANIFEST,
        pool=pool,
        source_run_id=result.run_dir.name,
    )
    folds = range(MANIFEST.num_folds)
    assert [v.path for v in verified.development] == [
        result.artifacts[f"fold{k}_fixed_budget"] for k in folds
    ]
    assert [v.provenance["source_epoch"] for v in verified.selected] == list(PEAKS)
    assert verified.refit.temperature == pytest.approx(result.refit_temperature.temperature)
    assert all(v.temperature is None for v in verified.development)
    assert isinstance(verified.refit.model, torch.nn.Module)


@pytest.mark.parametrize(
    ("artifact", "overrides", "message"),
    [
        # Own-best and all-32 substitution for a fold's development encoder.
        ("fold2_best", {}, "role is 'selected_fold_analysis'"),
        ("refit", {}, "role is 'all_32_refit'"),
        ("fold3_fixed_budget", {}, "fold_index is 3"),
        ("fold2_fixed_budget", {"budget": BUDGET + 1}, "common_budget is 16"),
        ("fold2_fixed_budget", {"horizon": ESCALATED_HORIZON_EPOCHS}, "accepted_horizon is 30"),
        ("fold2_fixed_budget", {"lineage": cv.CANONICAL}, "must record a CUDA device"),
        ("fold2_fixed_budget", {"sha256": "0" * 64}, "the consumer references"),
        ("fold2_fixed_budget", {"source_run_id": "another_run"}, "source_run_id is"),
    ],
)
def test_the_verifier_refuses_the_wrong_artifact(run, pool, artifact, overrides, message) -> None:
    """Each file is self-consistent; only the consumer's own expectation exposes it."""
    result, _ = run
    expected = _expect(cv.COMMON_BUDGET_DEVELOPMENT, fold=2, **overrides)
    with pytest.raises(fa.F1ArtifactError, match=message):
        fa.verify_f1_artifact(result.artifacts[artifact], expected, pool=pool)


def test_best_pt_is_checked_against_its_own_epoch_not_b(run, pool) -> None:
    result, _ = run
    path = result.artifacts["fold2_best"]
    assert fa.verify_f1_artifact(path, _expect(cv.SELECTED_FOLD_ANALYSIS, fold=2), pool=pool)
    wrong = _expect(cv.SELECTED_FOLD_ANALYSIS, fold=2, selected_epoch=BUDGET)
    with pytest.raises(fa.F1ArtifactError, match=f"source_epoch is {PEAKS[2]}"):
        fa.verify_f1_artifact(path, wrong, pool=pool)


def _fold_config(result: cv.SubjectCVResult, k: int) -> dict[str, Any]:
    return load_checkpoint(result.artifacts[f"fold{k}_fixed_budget"])[1]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c, other: c["provenance"].pop("class_weights_sha256"), "missing.*class_weights"),
        # Not another fold's digest: every synthetic subject has the same label mix, so all folds'
        # weights coincide here. Uniform weights are a vector D11 never produces for this pool.
        (
            lambda c, other: c["provenance"].update(
                class_weights_sha256=class_weights_sha256(np.ones(18, dtype=np.float32))
            ),
            "class_weights_sha256 is",
        ),
        (
            lambda c, other: c["provenance"].update(
                training_subject_ids=other["provenance"]["training_subject_ids"]
            ),
            "training_subject_ids is",
        ),
        (
            lambda c, other: c["provenance"].update(scoring_subject_ids=[1, 2, 3, 5]),
            "final-test subject",
        ),
        (lambda c, other: c["provenance"].pop("source_window_identity_sha256"), "window_identity"),
        (
            lambda c, other: c.update(probability_calibration={"status": "fitted"}),
            "must carry no temperature",
        ),
    ],
)
def test_the_verifier_refuses_tampered_provenance(run, pool, tmp_path, edit, message) -> None:
    result, _ = run
    other = _fold_config(result, 3)
    path = _rewrite(
        result.artifacts["fold2_fixed_budget"], tmp_path / "fixed.pt", lambda c: edit(c, other)
    )
    with pytest.raises(fa.F1ArtifactError, match=message):
        fa.verify_f1_artifact(path, _expect(cv.COMMON_BUDGET_DEVELOPMENT, fold=2), pool=pool)


def test_normalization_is_recomputed_not_trusted(run, pool, tmp_path) -> None:
    """Statistics from another pool, with a matching digest, still fail against the refit."""
    from temporallens.utils.run_logger import config_digest

    result, _ = run
    other = _fold_config(result, 3)["normalization"]

    def swap(config: dict[str, Any]) -> None:
        config["normalization"] = other
        config["provenance"]["normalization_sha256"] = config_digest(other)

    path = _rewrite(result.artifacts["fold2_fixed_budget"], tmp_path / "fixed.pt", swap)
    with pytest.raises(fa.F1ArtifactError, match="normalization mean does not match"):
        fa.verify_f1_artifact(path, _expect(cv.COMMON_BUDGET_DEVELOPMENT, fold=2), pool=pool)


def test_a_different_windowing_is_caught_by_the_window_identity(run, data_dir) -> None:
    result, _ = run
    restrided = cv.load_development_pool(_config(data_dir, stride=100))
    with pytest.raises(fa.F1ArtifactError, match="source_window_identity_sha256 is"):
        fa.verify_f1_artifact(
            result.artifacts["fold2_fixed_budget"],
            _expect(cv.COMMON_BUDGET_DEVELOPMENT, fold=2),
            pool=restrided,
        )


def test_a_refit_must_not_claim_a_fold(run, pool, tmp_path) -> None:
    result, _ = run
    path = _rewrite(
        result.artifacts["refit"],
        tmp_path / "refit.pt",
        lambda c: c["provenance"].update(fold_index=0),
    )
    with pytest.raises(fa.F1ArtifactError, match="must not record fold_index"):
        fa.verify_f1_artifact(path, _expect(cv.ALL_32_REFIT), pool=pool)


def test_the_refit_temperature_must_name_this_runs_development_models(run, pool, tmp_path) -> None:
    """Every artifact can pass alone while the refit's T came from other development models."""
    result, base = run
    copy = tmp_path / "checkpoints"
    shutil.copytree(base / "checkpoints", copy)
    refit = refit_checkpoint_path(cv.lineage_root(copy, cv.PRELIMINARY), "f1_test")

    def reorder(config: dict[str, Any]) -> None:
        digests = config["probability_calibration"]["fold_artifact_sha256"]
        config["probability_calibration"]["fold_artifact_sha256"] = digests[::-1]

    _rewrite(refit, refit, reorder)
    with pytest.raises(fa.F1ArtifactError, match="not from this run's development models"):
        fa.verify_f1_run(
            copy,
            "f1_test",
            selected_epochs=PEAKS,
            horizon=INITIAL_HORIZON_EPOCHS,
            lineage=cv.PRELIMINARY,
            manifest=MANIFEST,
            pool=pool,
        )


def test_missing_or_provenance_free_files_are_refused_but_still_load_generically(
    run, pool, tmp_path
) -> None:
    """The generic loader stays protocol-agnostic (F0); only the F1 verifier needs provenance."""
    result, _ = run
    expected = _expect(cv.COMMON_BUDGET_DEVELOPMENT, fold=2)
    with pytest.raises(fa.F1ArtifactError, match="missing F1 artifact"):
        fa.verify_f1_artifact(tmp_path / "absent.pt", expected, pool=pool)
    bare = _rewrite(
        result.artifacts["fold2_fixed_budget"], tmp_path / "bare.pt", lambda c: c.pop("provenance")
    )
    assert isinstance(rebuild_model(bare), torch.nn.Module)
    with pytest.raises(fa.F1ArtifactError, match="carries no provenance"):
        fa.verify_f1_artifact(bare, expected, pool=pool)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"role": "best"}, "unknown F1 artifact role"),
        ({"role": cv.COMMON_BUDGET_DEVELOPMENT}, "needs a fold"),
        ({"role": cv.ALL_32_REFIT, "fold": 0}, "belongs to no fold"),
        ({"role": cv.SELECTED_FOLD_ANALYSIS, "fold": 0}, "own selected epoch"),
        ({"role": cv.COMMON_BUDGET_DEVELOPMENT, "fold": 0, "selected_epoch": 12}, "only best.pt"),
        ({"role": cv.ALL_32_REFIT, "budget": INITIAL_HORIZON_EPOCHS}, "cannot be accepted"),
        ({"role": cv.ALL_32_REFIT, "horizon": 45}, "30 or 60"),
    ],
)
def test_an_impossible_expectation_is_refused(fields, message) -> None:
    values: dict[str, Any] = {
        "budget": BUDGET,
        "horizon": INITIAL_HORIZON_EPOCHS,
        "lineage": cv.PRELIMINARY,
        "manifest": MANIFEST,
        **fields,
    }
    with pytest.raises(ValueError, match=message):
        fa.F1Expectation(**values)


def test_the_verifier_requires_every_key_the_f1_config_declares() -> None:
    declared = _yaml(F1_CONFIG)["training"]["development_checkpoints"]
    for role in fa.FOLD_ROLES:
        assert set(declared["required_source_provenance"]) <= set(fa.REQUIRED_PROVENANCE[role])


def test_the_local_language_smoke_test_freezes_the_preliminary_f1_refit() -> None:
    """L1 runs before any CUDA F1 exists, so it reads the location this harness writes (D47)."""
    experiment = _yaml(F1_CONFIG)["experiment"]["name"]
    preliminary = refit_checkpoint_path(cv.lineage_root("checkpoints", cv.PRELIMINARY), experiment)
    l1 = _yaml(REPO_ROOT / "configs/experiment/language/adapter_llama1b_local.yaml")
    assert Path(l1["encoder"]["checkpoint"]) == preliminary
    assert Path(l1["encoder"]["checkpoint"]) != refit_checkpoint_path("checkpoints", experiment)

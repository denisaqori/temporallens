"""The config-to-run-config bridge: consume every key or refuse.

These tests exist because the opposite behaviour shipped three times. `loss` was declared in
every foundation config and read by nothing; so were `device`, `model.type` and
`dataset.normalize`. Each read as a commitment the code did not honour. The loader's job is to
make that impossible, so most of what follows checks that it *refuses* rather than that it works.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from _synthetic import recordings

from temporallens.data.ninapro import processed_path, save_processed
from temporallens.training.encoder import EncoderRunConfig, train_encoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_CONFIG = REPO_ROOT / "configs/experiment/foundation/debug_tiny.yaml"


@functools.lru_cache(maxsize=1)
def _script() -> ModuleType:
    """Loaded once. Re-executing the module would mint a fresh `ConfigError` class each call, and
    `pytest.raises` would then never match the one actually raised."""
    spec = importlib.util.spec_from_file_location(
        "temporallens_train_encoder_script", REPO_ROOT / "scripts" / "train_encoder.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def debug_config() -> dict:
    return yaml.safe_load(DEBUG_CONFIG.read_text())


# --- the contract stays in step with the dataclass --------------------------------------------


def test_field_paths_cover_every_run_config_field() -> None:
    """The meta-test that keeps the loader honest.

    A field added to `EncoderRunConfig` with no config path is a parameter no config can set; a
    path naming a field that no longer exists is dead. `output_dir` is deliberately absent — it is
    a CLI argument, because where runs land is not a protocol choice.
    """
    script = _script()
    fields = {f.name for f in dataclasses.fields(EncoderRunConfig)}
    mapped = set(script.FIELD_PATHS.values())
    assert (
        mapped <= fields
    ), f"FIELD_PATHS names fields that do not exist: {sorted(mapped - fields)}"
    assert fields - mapped == {
        "output_dir"
    }, f"run-config fields no config key can set: {sorted(fields - mapped - {'output_dir'})}"


def test_produced_metrics_matches_what_a_run_actually_computes(tmp_path) -> None:
    """`PRODUCED_METRICS` gates what a config may ask for, so a stale list would lie twice.

    Too narrow and it refuses a metric the run does compute; too wide and the refusal it exists
    to raise never fires.
    """
    data_dir = tmp_path / "processed"
    for subject, rec in recordings((1, 2, 3), gestures=3, repetitions=2).items():
        save_processed(rec, processed_path(data_dir, subject))
    result = train_encoder(
        EncoderRunConfig(
            name="metric_parity",
            seed=42,
            data_dir=data_dir,
            output_dir=tmp_path / "out",
            train_subjects=[1, 2],
            held_out_subjects=[3],
            window_size=400,
            stride=200,
            max_windows_per_subject=60,
            input_channels=12,
            num_classes=18,
            hidden_dim=8,
            embedding_dim=16,
            dropout=0.1,
            batch_size=8,
            epochs=1,
            learning_rate=1e-3,
        )
    )
    computed = set(result.metrics) | set(result.per_subject_metrics)
    assert computed == set(_script().PRODUCED_METRICS)


# --- it loads the real F0 config ---------------------------------------------------------------


def test_the_real_debug_config_loads_and_maps_every_value() -> None:
    script = _script()
    config, recorded = script.load_run_config(DEBUG_CONFIG, Path("results/runs"))
    assert (config.train_subjects, config.held_out_subjects) == ([1, 2], [3])
    assert config.model_type == "cnn1d"
    assert config.normalize == "train_subject_covered_sample_stats"
    assert config.device == "auto"
    assert config.weight_decay == 0.0001, "F0 must declare it, not inherit the 0.0 default"
    assert recorded["experiment.mode"] == "debug"


# --- and refuses everything it cannot honour exactly -------------------------------------------


def test_an_unconsumed_key_is_refused(tmp_path, debug_config) -> None:
    debug_config["training"]["momentum"] = 0.9
    with pytest.raises(_script().ConfigError, match="does not consume"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


def test_a_missing_required_key_is_refused(tmp_path, debug_config) -> None:
    del debug_config["training"]["learning_rate"]
    with pytest.raises(_script().ConfigError, match="missing required keys"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


def test_an_optional_key_may_be_absent(tmp_path, debug_config) -> None:
    del debug_config["dataset"]["max_windows_per_subject"]
    config, _ = _script().load_run_config(_write(tmp_path, debug_config), tmp_path)
    assert config.max_windows_per_subject is None


def test_a_cross_validated_config_is_refused_rather_than_run_as_one_split() -> None:
    """A single-split number from F1's config would look like F1 and not be it."""
    script = _script()
    path = REPO_ROOT / "configs/experiment/foundation/baseline_cnn_subject_split.yaml"
    with pytest.raises(script.ConfigError, match="split_manifest"):
        script.load_run_config(path, Path("results/runs"))


def test_the_random_window_config_is_refused_by_a_subject_split_path() -> None:
    script = _script()
    path = REPO_ROOT / "configs/experiment/foundation/baseline_cnn_random_split.yaml"
    with pytest.raises(script.ConfigError, match="leakage demonstration"):
        script.load_run_config(path, Path("results/runs"))


def test_a_checked_key_with_the_wrong_value_is_refused(tmp_path, debug_config) -> None:
    debug_config["dataset"]["exercise"] = "A"
    with pytest.raises(_script().ConfigError, match="must be 'B'"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


def test_online_tracking_is_refused_because_nothing_logs_remotely(tmp_path, debug_config) -> None:
    debug_config["tracking"]["wandb_mode"] = "online"
    with pytest.raises(_script().ConfigError, match="wandb_mode"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


def test_mirrored_keys_that_disagree_are_refused(tmp_path, debug_config) -> None:
    """`input_channels` is declared under both `dataset` and `model`. One of them is decoration
    unless they are checked against each other, and nobody would know which was used."""
    debug_config["model"]["input_channels"] = 8
    with pytest.raises(_script().ConfigError, match="must agree"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


def test_a_declared_metric_the_run_never_computes_is_refused(tmp_path, debug_config) -> None:
    debug_config["evaluation"]["metrics"].append("per_subject_brier_score")
    with pytest.raises(_script().ConfigError, match="does not compute"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


# --- strict reading: no silent last-write-wins, no silently optional checked keys --------------


def test_a_repeated_key_in_the_config_is_refused(tmp_path) -> None:
    """A second `epochs:` used to win silently: a config showing 2 near the top trained for 500."""
    text = DEBUG_CONFIG.read_text().replace("  epochs: 2\n", "  epochs: 2\n  epochs: 500\n")
    assert text.count("epochs:") == 2, "premise: the duplicate was inserted"
    path = tmp_path / "duplicate.yaml"
    path.write_text(text)
    with pytest.raises(_script().ConfigError, match="duplicate YAML mapping key: 'epochs'"):
        _script().load_run_config(path, tmp_path)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("dataset", "name"),
        ("dataset", "exercise"),
        ("tracking", "json_logging"),
        ("experiment", "mode"),
    ],
)
def test_a_checked_key_may_not_be_omitted(tmp_path, debug_config, section, key) -> None:
    """A checked key that is only checked when present can be dropped to dodge the check."""
    del debug_config[section][key]
    with pytest.raises(_script().ConfigError, match="missing required keys"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


def test_a_mode_this_script_does_not_run_is_refused(tmp_path, debug_config) -> None:
    debug_config["experiment"]["mode"] = "evaluate"
    with pytest.raises(_script().ConfigError, match="experiment.mode"):
        _script().load_run_config(_write(tmp_path, debug_config), tmp_path)


def test_the_script_records_its_metadata_in_the_run_log(tmp_path, debug_config) -> None:
    """What the loader records -- mode, protocol state, tracking mode, declared metrics -- and the
    exact config file it read must reach run.json, not stop at the loader."""
    import hashlib
    import json

    from temporallens.data.ninapro import processed_path, save_processed

    data_dir = tmp_path / "processed"
    for subject, rec in recordings((1, 2, 3), gestures=3, repetitions=2).items():
        save_processed(rec, processed_path(data_dir, subject))
    debug_config["dataset"]["data_dir"] = str(data_dir)
    config_path = _write(tmp_path, debug_config)

    assert (
        _script().main(["--config", str(config_path), "--output-dir", str(tmp_path / "runs")]) == 0
    )
    (run_dir,) = (tmp_path / "runs").iterdir()
    metadata = json.loads((run_dir / "run.json").read_text())["metadata"]
    assert metadata["experiment.mode"] == "debug"
    assert metadata["tracking.wandb_mode"] == "disabled"
    assert metadata["declared_metrics"] == sorted(debug_config["evaluation"]["metrics"])
    assert metadata["config_path"] == str(config_path)
    assert metadata["config_file_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()


def test_a_non_debug_run_from_uncommitted_code_is_refused(
    tmp_path, debug_config, monkeypatch
) -> None:
    """Only a debug run may start from uncommitted code; anything else could become a number."""
    from temporallens.data.ninapro import processed_path, save_processed
    from temporallens.utils import run_logger

    data_dir = tmp_path / "processed"
    for subject, rec in recordings((1, 2, 3), gestures=3, repetitions=2).items():
        save_processed(rec, processed_path(data_dir, subject))
    debug_config["dataset"]["data_dir"] = str(data_dir)
    debug_config["experiment"]["mode"] = "train"
    config_path = _write(tmp_path, debug_config)
    dirty = {"commit": "a" * 40, "dirty": True, "changes": [" M x.py"], "diff_sha256": "f" * 64}
    monkeypatch.setattr(run_logger, "get_git_state", lambda: dict(dirty))

    with pytest.raises(SystemExit) as exit_info:
        _script().main(["--config", str(config_path), "--output-dir", str(tmp_path / "runs")])
    assert exit_info.value.code == 2
    assert not (tmp_path / "runs").exists(), "the refusal must come before any run directory"


def test_the_clean_tree_refusal_comes_before_any_data_is_loaded(tmp_path, monkeypatch) -> None:
    """For F1 that is 32 subjects; refusing after loading them all is a slow way to say no."""
    from temporallens.training import encoder
    from temporallens.utils import run_logger

    dirty = {"commit": "a" * 40, "dirty": True, "changes": [" M x.py"], "diff_sha256": "f" * 64}
    monkeypatch.setattr(run_logger, "get_git_state", lambda: dict(dirty))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("data was loaded before the clean-tree check")

    monkeypatch.setattr(encoder, "_load", fail_if_called)
    config, _ = _script().load_run_config(DEBUG_CONFIG, tmp_path)
    with pytest.raises(run_logger.UncommittedCodeError):
        encoder.train_encoder(config, require_clean_tree=True)


# --- F1: the cross-validation harness bridge -------------------------------------------------

F1_CONFIG = REPO_ROOT / "configs/experiment/foundation/baseline_cnn_subject_split.yaml"


@pytest.fixture
def f1_config() -> dict:
    return yaml.safe_load(F1_CONFIG.read_text())


def test_the_real_f1_config_builds_the_harness_config() -> None:
    script = _script()
    config, recorded = script.build_f1_config(script.read_config(F1_CONFIG), F1_CONFIG)
    assert config.name == "baseline_cnn_subject_split"
    assert config.manifest.num_folds == 8 and len(config.manifest.training_subjects) == 32
    assert (config.learning_rate, config.weight_decay, config.batch_size) == (3e-4, 1e-4, 64)
    assert (config.window_size, config.stride, config.hidden_dim) == (400, 100, 128)
    assert recorded["final_test_declarations"] == "checked_not_run"


def test_every_f1_key_is_mapped_checked_declared_or_recorded() -> None:
    """The F1 meta-test: each leaf of the real config belongs to exactly one table."""
    script = _script()
    leaves = set(script.leaf_paths(script.read_config(F1_CONFIG)))
    tables = [
        set(script.F1_FIELD_PATHS),
        set(script.F1_CHECKED_PATHS),
        set(script.F1_FINAL_TEST_PATHS),
        set(script.F1_RECORDED_PATHS),
        {mirror for _, mirror in script.F1_MIRRORED_PATHS},
        {script.F1_MANIFEST_PATH, "evaluation.temperature_fit.checkpoint_pattern"},
    ]
    assert set().union(*tables) == leaves
    assert sum(len(t) for t in tables) == len(leaves)  # no key in two tables


def test_f1_field_paths_cover_every_harness_config_field() -> None:
    from temporallens.training.subject_cv import SubjectCVConfig

    fields = {f.name for f in dataclasses.fields(SubjectCVConfig)} - {"manifest"}
    assert set(_script().F1_FIELD_PATHS.values()) == fields


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (("training", "epoch_selection"), "smoothing_window_epochs", 5),
        (("training", "optimization"), "betas", [0.9, 0.99]),
        (("training", "optimization"), "amsgrad", 0),  # YAML 0 is not false
        (("training", "horizon_escalation"), "next_fold_epochs", 90),
        (("evaluation", "temperature_fit", "execution"), "bounds", [0.001, 1000.0]),
        (("evaluation", "ece"), "bins", 15),  # a final-test declaration
        (("training",), "epochs", 20),
    ],
)
def test_a_changed_f1_protocol_value_is_refused(tmp_path, f1_config, section, key, value) -> None:
    node = f1_config
    for part in section:
        node = node[part]
    node[key] = value
    with pytest.raises(_script().ConfigError, match=key):
        _script().build_f1_config(f1_config, _write(tmp_path, f1_config))


def test_an_unconsumed_f1_key_is_refused(tmp_path, f1_config) -> None:
    f1_config["training"]["early_stopping"] = False
    with pytest.raises(_script().ConfigError, match="does not consume"):
        _script().build_f1_config(f1_config, _write(tmp_path, f1_config))


def test_a_missing_f1_key_is_refused(tmp_path, f1_config) -> None:
    del f1_config["training"]["epoch_selection"]["tie_break"]
    with pytest.raises(_script().ConfigError, match="missing required keys"):
        _script().build_f1_config(f1_config, _write(tmp_path, f1_config))


def test_a_development_pattern_other_than_the_written_path_is_refused(tmp_path, f1_config) -> None:
    f1_config["evaluation"]["temperature_fit"][
        "checkpoint_pattern"
    ] = "elsewhere/fold{fold_index}.pt"
    with pytest.raises(_script().ConfigError, match="the path the harness writes"):
        _script().build_f1_config(f1_config, _write(tmp_path, f1_config))


def _f1_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--config",
        str(F1_CONFIG),
        "--output-dir",
        str(tmp_path / "runs"),
        "--checkpoint-root",
        str(tmp_path / "checkpoints"),
        *extra,
    ]


def test_main_routes_an_f1_config_to_the_harness_as_a_preliminary_run(
    tmp_path, monkeypatch
) -> None:
    """The wiring only: F1 goes to the harness as preliminary, still requiring a clean tree."""
    script = _script()
    seen: dict = {}

    class Done(Exception):
        pass

    def harness(config, **kwargs):
        seen.update(config=config, **kwargs)
        raise Done

    monkeypatch.setattr(script, "run_subject_cross_validation", harness)
    with pytest.raises(Done):
        script.main(_f1_args(tmp_path))
    assert seen["scope"] == "preliminary"  # never inferred from CUDA or a clean tree
    assert seen["require_clean_tree"] is True
    assert seen["checkpoint_base"] == tmp_path / "checkpoints"
    assert seen["config"].name == "baseline_cnn_subject_split"
    assert seen["metadata"]["config_file_sha256"]


def test_main_refuses_a_reportable_f1_run_before_loading_data(tmp_path) -> None:
    """--reportable reaches the real harness, which refuses before any run directory exists."""
    with pytest.raises(SystemExit) as exit_info:
        _script().main(_f1_args(tmp_path, "--reportable"))
    assert exit_info.value.code == 2
    assert not (tmp_path / "runs").exists() and not (tmp_path / "checkpoints").exists()


def test_reportable_is_refused_for_an_f0_config(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _script().main(["--config", str(DEBUG_CONFIG), "--reportable"])

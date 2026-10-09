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

"""The checkpoint contract: `{model_state, model_config}`, rebuildable from the file alone.

AGENTS.md states it as a hard constraint and `robustness_targets.yaml` depends on it -- a target
is just `(name, artifact)` with no `model_type`, which only works if the config travels inside
the checkpoint. These tests are that contract, not a serialisation smoke test.
"""

from __future__ import annotations

import pytest
import torch
from temporallens.models.checkpoint import (
    CheckpointError,
    fold_checkpoint_path,
    load_checkpoint,
    rebuild_model,
    refit_checkpoint_path,
    save_checkpoint,
)
from temporallens.models.encoders.cnn1d import Cnn1dClassifier

F1 = {
    "input_channels": 12,
    "hidden_dim": 128,
    "embedding_dim": 256,
    "num_classes": 18,
    "dropout": 0.2,
}


def _stats_payload():
    return {
        "mean": [0.0] * 12,
        "std": [1.0] * 12,
        "fitted_on": "fold_0_training_28",
        "num_samples": 1234,
    }


# --- the contract --------------------------------------------------------------------------


def test_a_checkpoint_carries_exactly_state_and_config(tmp_path) -> None:
    model = Cnn1dClassifier(**F1)
    path = tmp_path / "refit.pt"
    save_checkpoint(path, model, normalization=_stats_payload())

    raw = torch.load(path, map_location="cpu", weights_only=False)
    assert set(raw) == {"model_state", "model_config"}, (
        "the contract is two keys; extra top-level keys are how consumers start depending on "
        "things the contract does not promise"
    )


def test_a_consumer_rebuilds_from_the_file_with_no_external_model_type(tmp_path) -> None:
    """The reason robustness targets are just `(name, artifact)`."""
    model = Cnn1dClassifier(**F1).eval()
    path = tmp_path / "refit.pt"
    save_checkpoint(path, model, normalization=_stats_payload())

    rebuilt = rebuild_model(path).eval()
    x = torch.randn(2, 12, 400)
    torch.testing.assert_close(model(x), rebuilt(x))


def test_normalization_statistics_travel_with_the_checkpoint(tmp_path) -> None:
    """DESIGN DECISION UNDER TEST (needs owner review).

    The encoder is frozen and reused by both arms. If its statistics do not travel with the
    file, each arm normalizes differently and the F1 reference row stops being comparable --
    the same argument D23 makes for the temperature.
    """
    model = Cnn1dClassifier(**F1)
    path = tmp_path / "refit.pt"
    save_checkpoint(path, model, normalization=_stats_payload())

    _, config = load_checkpoint(path)
    assert config["normalization"]["fitted_on"] == "fold_0_training_28"
    assert len(config["normalization"]["mean"]) == 12


def test_saving_without_normalization_is_refused(tmp_path) -> None:
    """A checkpoint whose statistics are missing is unusable downstream, so fail at write time."""
    model = Cnn1dClassifier(**F1)
    with pytest.raises(CheckpointError, match="normalization"):
        save_checkpoint(tmp_path / "refit.pt", model, normalization=None)  # type: ignore[arg-type]


def test_a_checkpoint_missing_model_config_is_refused(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": Cnn1dClassifier(**F1).state_dict()}, path)
    with pytest.raises(CheckpointError, match="model_config"):
        load_checkpoint(path)


def test_an_unknown_model_type_is_refused_by_name(tmp_path) -> None:
    path = tmp_path / "odd.pt"
    torch.save(
        {"model_state": {}, "model_config": {"type": "transformer_xl", "normalization": {}}}, path
    )
    with pytest.raises(CheckpointError, match="transformer_xl"):
        rebuild_model(path)


def test_round_trip_preserves_every_parameter(tmp_path) -> None:
    torch.manual_seed(3)
    model = Cnn1dClassifier(**F1)
    path = tmp_path / "refit.pt"
    save_checkpoint(path, model, normalization=_stats_payload())

    state, _ = load_checkpoint(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(state[key], value)


# --- the two filenames D7 distinguishes -----------------------------------------------------


def test_the_two_checkpoint_kinds_have_distinct_paths(tmp_path) -> None:
    """D7: `refit.pt` is what downstream reads; fold checkpoints are analysis only."""
    assert refit_checkpoint_path(tmp_path, "baseline_cnn_subject_split").name == "refit.pt"

    fold = fold_checkpoint_path(tmp_path, "baseline_cnn_subject_split", fold=3)
    assert fold.name == "best.pt"
    assert fold.parent.name == "fold3"
    assert "folds" in fold.parts

    assert refit_checkpoint_path(tmp_path, "x") != fold_checkpoint_path(tmp_path, "x", fold=0)


def test_fold_paths_are_distinct_per_fold(tmp_path) -> None:
    paths = {fold_checkpoint_path(tmp_path, "run", fold=k) for k in range(8)}
    assert len(paths) == 8

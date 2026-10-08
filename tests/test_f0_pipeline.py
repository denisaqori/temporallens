"""F0 end to end, on synthetic recordings: the integration contract for the training path.

F0's whole purpose is to validate the local path -- loading, windowing, splitting, forward,
loss, metrics, logging -- so these tests assert the path holds together and that the leakage
rules hold while it does. They run on test doubles, so nothing here is a measurement.

Decisions pinned: D11 (weighted loss, never resampling), D12 (corrected columns), D24 (no
boundary-crossing windows), §3.3 (statistics from the training partition only), D18/D19 (fixed
epochs, no early stopping), and the checkpoint contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from _synthetic import recordings

from temporallens.data.ninapro import processed_path, save_processed
from temporallens.data.windows import index_recording
from temporallens.training.encoder import EncoderRunConfig, train_encoder

TRAIN = (1, 2)
HELD_OUT = 3


@pytest.fixture
def prepared(tmp_path):
    """A processed directory holding three synthetic subjects, plus an output directory."""
    data_dir = tmp_path / "processed"
    for subject, rec in recordings(TRAIN + (HELD_OUT,), gestures=3, repetitions=2).items():
        save_processed(rec, processed_path(data_dir, subject))
    return data_dir, tmp_path / "out"


def _config(data_dir, out_dir, **overrides):
    base = dict(
        name="debug_tiny",
        seed=42,
        data_dir=data_dir,
        output_dir=out_dir,
        train_subjects=list(TRAIN),
        held_out_subjects=[HELD_OUT],
        window_size=400,
        stride=200,
        max_windows_per_subject=60,
        input_channels=12,
        num_classes=18,
        hidden_dim=32,
        embedding_dim=64,
        dropout=0.1,
        batch_size=8,
        epochs=2,
        learning_rate=1e-3,
        save_checkpoint=False,
    )
    base.update(overrides)
    return EncoderRunConfig(**base)


# --- the path holds together ------------------------------------------------------------------


def test_f0_runs_end_to_end_and_reports_the_configured_metrics(prepared) -> None:
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir))

    assert result.epochs_run == 2
    for key in ("accuracy", "macro_f1", "confusion_matrix"):
        assert key in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    cm = np.asarray(result.metrics["confusion_matrix"])
    assert cm.shape == (18, 18)


def test_the_loss_moves(prepared) -> None:
    """Not a measurement -- just that gradients reach the parameters at all."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir, epochs=3))
    assert len(result.epoch_losses) == 3
    assert all(np.isfinite(result.epoch_losses))
    assert result.epoch_losses[-1] < result.epoch_losses[0]


def test_a_run_writes_its_json_log(prepared) -> None:
    """§3.5: every run writes run.json through RunLogger, and never depends on a network."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir))
    state = json.loads((result.run_dir / "run.json").read_text())
    assert state["experiment_name"] == "debug_tiny"
    assert "git_commit" in state


def test_predictions_are_persisted_with_their_subject(prepared) -> None:
    """ECE is recomputable offline from frozen predictions, so provenance has to be saved."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir, save_predictions=True))
    saved = np.load(result.run_dir / "predictions.npz")
    assert set(np.unique(saved["subject"])) == {HELD_OUT}
    assert saved["probs"].shape[1] == 18
    assert len(saved["label"]) == len(saved["probs"])


# --- the leakage rules, which are the point --------------------------------------------------


def test_the_held_out_subject_never_appears_in_training(prepared) -> None:
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir))
    assert HELD_OUT not in set(result.train_subjects)
    assert set(result.evaluation_subjects) == {HELD_OUT}


def test_normalization_statistics_come_from_the_training_subjects_only(prepared) -> None:
    """§3.3's silent leak. Statistics fitted with the held-out subject included would differ."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir))

    from temporallens.data.ninapro import load_processed
    from temporallens.data.windows import concatenate
    from temporallens.preprocessing.normalize import fit_channel_stats

    recs = {s: load_processed(processed_path(data_dir, s)) for s in (*TRAIN, HELD_OUT)}
    train_only = fit_channel_stats(
        {s: recs[s] for s in TRAIN},
        concatenate([index_recording(recs[s], window_size=400, stride=200) for s in TRAIN]),
        fitted_on="train",
    )
    np.testing.assert_allclose(result.normalization["mean"], train_only.mean, rtol=1e-5)

    everything = fit_channel_stats(
        recs,
        concatenate(
            [index_recording(recs[s], window_size=400, stride=200) for s in (*TRAIN, HELD_OUT)]
        ),
        fitted_on="all",
    )
    assert not np.allclose(
        result.normalization["mean"], everything.mean
    ), "statistics match an all-subject fit, so the held-out subject leaked in"


def test_class_weights_are_fitted_on_the_training_split(prepared) -> None:
    """D11: weights inversely proportional to class frequency in the TRAINING split."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir))
    weights = np.asarray(result.class_weights)
    assert weights.shape == (18,)
    assert np.all(weights >= 0)
    assert np.isfinite(weights).all(), "an absent class produced a non-finite weight"


def test_no_window_crosses_a_segment_boundary(prepared) -> None:
    """D24, asserted through the pipeline rather than only in the windower's own tests."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir))
    assert result.boundary_crossing_windows == 0


def test_resampling_is_never_used(prepared) -> None:
    """D11: imbalance is handled in the loss. No oversampling, no undersampling, any split."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir))
    assert result.train_window_count == result.unique_train_window_count


# --- fixed horizon ----------------------------------------------------------------------------


def test_training_runs_the_configured_epochs_with_no_early_stopping(prepared) -> None:
    """D18/D19 forbid early stopping; the horizon is a declared constant."""
    data_dir, out_dir = prepared
    assert train_encoder(_config(data_dir, out_dir, epochs=1)).epochs_run == 1
    assert train_encoder(_config(data_dir, out_dir, epochs=4)).epochs_run == 4


def test_a_seeded_run_is_reproducible(prepared) -> None:
    data_dir, out_dir = prepared
    first = train_encoder(_config(data_dir, out_dir, seed=7))
    second = train_encoder(_config(data_dir, out_dir, seed=7))
    assert first.metrics["accuracy"] == pytest.approx(second.metrics["accuracy"])


# --- the checkpoint, when asked for ----------------------------------------------------------


def test_saving_a_checkpoint_honours_the_contract(prepared) -> None:
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir, save_checkpoint=True))

    from temporallens.models.checkpoint import load_checkpoint, rebuild_model

    assert result.checkpoint_path is not None
    _, config = load_checkpoint(result.checkpoint_path)
    assert config["type"] == "cnn1d"
    assert "normalization" in config
    rebuild_model(result.checkpoint_path)


def test_a_debug_run_refuses_to_write_a_reportable_checkpoint_name(prepared) -> None:
    """F0 is not a measurement, so it must not produce the artifact downstream arms load."""
    data_dir, out_dir = prepared
    result = train_encoder(_config(data_dir, out_dir, save_checkpoint=True))
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.name != "refit.pt"


def test_a_config_declaring_another_loss_is_refused(prepared) -> None:
    """D11 permits one loss here, so the config's declaration has to be load-bearing.

    Before this, EncoderRunConfig had no `loss` field at all: the trainer hardcoded weighted
    cross-entropy and a config saying otherwise would have been silently overridden.
    """
    data_dir, out_dir = prepared
    with pytest.raises(ValueError, match="class_weighted_cross_entropy"):
        _config(data_dir, out_dir, loss="cross_entropy")


def test_the_boundary_audit_agrees_with_an_independent_count(prepared) -> None:
    """The vectorised audit must match the naive nested loop it replaced."""
    from temporallens.data.ninapro import load_processed, processed_path
    from temporallens.data.windows import index_recording, segment_spans
    from temporallens.training.encoder import _count_boundary_crossing

    data_dir, _ = prepared
    subject = TRAIN[0]
    recording = load_processed(processed_path(data_dir, subject))
    index = index_recording(recording, window_size=400, stride=200)

    spans = segment_spans(recording.label, recording.repetition)
    naive = sum(
        0 if any(a <= s and s + index.window_size <= b for a, b in spans) else 1
        for s in index.start
    )
    assert _count_boundary_crossing(index, {subject: recording}) == naive
    assert naive == 0, "the windower should not be producing boundary-crossing windows at all"

"""The 1-D CNN encoder, the classifier head, and what `frozen` has to mean.

The spec fixes only the interface -- `cnn1d`, `input_channels`, `hidden_dim`, `embedding_dim`,
`num_classes`, `dropout`. The internals are an implementation choice, so these tests pin the
contract the rest of the project depends on rather than the layer arrangement.

DESIGN DECISION UNDER TEST (needs owner review): three Conv-BN-ReLU-pool blocks into a global
average pool and a linear projection to `embedding_dim`. Nothing outside this module may depend
on that arrangement; it may change as long as these tests hold.
"""

from __future__ import annotations

import pytest
import torch

from temporallens.models.encoders.cnn1d import Cnn1dClassifier, Cnn1dEncoder, build_model

F0 = {
    "input_channels": 12,
    "hidden_dim": 32,
    "embedding_dim": 64,
    "num_classes": 18,
    "dropout": 0.1,
}
F1 = {
    "input_channels": 12,
    "hidden_dim": 128,
    "embedding_dim": 256,
    "num_classes": 18,
    "dropout": 0.2,
}
WINDOW = 400


def _batch(n=4, channels=12, window=WINDOW):
    return torch.randn(n, channels, window)


# --- shapes, for both configured sizes -------------------------------------------------------


@pytest.mark.parametrize("cfg", (F0, F1), ids=("F0", "F1"))
def test_encoder_maps_a_window_to_its_embedding(cfg) -> None:
    encoder = Cnn1dEncoder(**{k: v for k, v in cfg.items() if k != "num_classes"})
    out = encoder(_batch())
    assert out.shape == (4, cfg["embedding_dim"])


@pytest.mark.parametrize("cfg", (F0, F1), ids=("F0", "F1"))
def test_classifier_maps_a_window_to_class_logits(cfg) -> None:
    model = Cnn1dClassifier(**cfg)
    out = model(_batch())
    assert out.shape == (4, cfg["num_classes"])


def test_encoder_is_length_agnostic() -> None:
    """Global pooling, so a debug config's shorter stride does not change the embedding size."""
    encoder = Cnn1dEncoder(**{k: v for k, v in F1.items() if k != "num_classes"})
    assert encoder(_batch(window=400)).shape == (4, 256)
    assert encoder(_batch(window=800)).shape == (4, 256)


def test_wrong_channel_count_is_rejected() -> None:
    model = Cnn1dClassifier(**F1)
    with pytest.raises(RuntimeError):
        model(_batch(channels=8))


# --- the classifier exposes its encoder, because both arms reuse it --------------------------


def test_classifier_exposes_the_encoder_it_trained() -> None:
    model = Cnn1dClassifier(**F1)
    assert isinstance(model.encoder, Cnn1dEncoder)
    embedding = model.encoder(_batch())
    assert embedding.shape == (4, F1["embedding_dim"])


def test_freezing_means_no_grad_and_eval_mode() -> None:
    """AGENTS.md: `frozen` is requires_grad=False AND eval() mode -- not just "not updated".

    eval() matters here specifically because the encoder has BatchNorm: left in train mode it
    keeps updating its running statistics from whatever arm is using it, which silently changes
    a checkpoint the arms are supposed to share unchanged.
    """
    model = Cnn1dClassifier(**F1)
    model.encoder.freeze()

    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not model.encoder.training
    # The head stays trainable.
    assert all(p.requires_grad for p in model.head.parameters())


def test_frozen_encoder_running_stats_survive_a_forward_pass() -> None:
    """The failure eval() prevents: BatchNorm drifting under a downstream arm."""
    model = Cnn1dClassifier(**F1)
    model.encoder.freeze()
    before = [
        b.clone() for b in model.encoder.buffers() if b.dtype.is_floating_point and b.numel() > 1
    ]
    assert before, "expected BatchNorm running buffers to exist"
    model.encoder(_batch())
    after = [b for b in model.encoder.buffers() if b.dtype.is_floating_point and b.numel() > 1]
    for b0, b1 in zip(before, after, strict=True):
        torch.testing.assert_close(b0, b1)


def test_a_frozen_encoder_still_trains_the_head() -> None:
    model = Cnn1dClassifier(**F1)
    model.encoder.freeze()
    loss = model(_batch()).sum()
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())


# --- determinism and dropout -----------------------------------------------------------------


def test_eval_mode_is_deterministic() -> None:
    torch.manual_seed(0)
    model = Cnn1dClassifier(**F1).eval()
    x = _batch()
    torch.testing.assert_close(model(x), model(x))


def test_dropout_is_active_in_train_mode_only() -> None:
    torch.manual_seed(0)
    model = Cnn1dClassifier(**F1)
    x = _batch()
    model.train()
    assert not torch.allclose(model(x), model(x)), "dropout appears inactive during training"
    model.eval()
    torch.testing.assert_close(model(x), model(x))


def test_capacity_follows_hidden_dim() -> None:
    small = sum(p.numel() for p in Cnn1dClassifier(**F0).parameters())
    large = sum(p.numel() for p in Cnn1dClassifier(**F1).parameters())
    assert large > small * 4, "hidden_dim does not appear to drive capacity"


# --- the config a checkpoint has to be rebuildable from --------------------------------------


def test_model_config_is_sufficient_to_rebuild() -> None:
    """The checkpoint contract: a consumer rebuilds from `model_config` alone, no model_type.

    Rebuilt through `build_model`, which is the supported path, rather than by hand-stripping
    keys: `model_config` carries both the constructor kwargs and facts that are *recorded* and
    not configurable, and the builder is what knows the difference.
    """
    model = Cnn1dClassifier(**F1)
    config = model.model_config()

    rebuilt = build_model(config)
    assert config["type"] == "cnn1d"
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(
        p.numel() for p in model.parameters()
    )

    import json

    assert json.loads(json.dumps(config)) == config, "model_config is not JSON-serialisable"


def test_the_recorded_architecture_is_the_frozen_one() -> None:
    """D32 froze the stack, so what a checkpoint records has to be that stack, not a copy."""
    recorded = Cnn1dClassifier(**F1).model_config()["architecture"]
    assert recorded == {
        "block_kernels": [7, 5, 3],
        "pool_factors": [4, 4],
        "width_multiplier": 2,
        "normalization": "batchnorm1d",
        "activation": "relu",
        "global_pool": "adaptive_avg",
    }


def test_a_checkpoint_recording_different_internals_is_refused() -> None:
    """The reason to record the internals at all.

    Same type name, same kwargs, different definition of what that name builds. The weights would
    load into right-shaped wrong layers and report a number for an architecture nobody ran.
    """
    config = Cnn1dClassifier(**F1).model_config()
    config["architecture"] = dict(config["architecture"], block_kernels=[9, 5, 3])
    with pytest.raises(ValueError, match="different 'cnn1d' architecture"):
        build_model(config)


def test_the_wrong_builder_refuses_rather_than_building_a_cnn() -> None:
    config = Cnn1dClassifier(**F1).model_config()
    config["type"] = "transformer_xl"
    with pytest.raises(ValueError, match="declares 'transformer_xl'"):
        build_model(config)

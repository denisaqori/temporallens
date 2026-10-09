"""Portable PyTorch device selection for local MPS and cloud CUDA runs."""

from __future__ import annotations

import torch

#: What `training.device` may say. `auto` is a request to pick; the others are requests to pin.
SUPPORTED_DEVICES = ("auto", "cpu", "mps", "cuda")


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_available(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    return torch.backends.mps.is_available()


def resolve_device(request: str) -> torch.device:
    """Turn a configured device request into the device a run will actually use.

    ``auto`` delegates to :func:`get_device`. Anything else is a pin, and an unavailable pin
    **raises** rather than falling back: a config that asks for ``cuda`` and silently gets ``cpu``
    produces a run whose recorded intent and actual behaviour disagree, which is the whole class
    of failure this field exists to prevent. The caller records the resolved device in the run
    log, so ``auto`` stays reproducible after the fact.
    """
    if request not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of {list(SUPPORTED_DEVICES)}, got {request!r}")
    if request == "auto":
        return get_device()
    if not _is_available(request):
        raise RuntimeError(
            f"device {request!r} was requested explicitly but is not available on this machine; "
            "use 'auto' to let the run pick, or change the config"
        )
    return torch.device(request)

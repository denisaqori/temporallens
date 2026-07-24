#!/usr/bin/env python3
"""Verify the TemporalLens local development environment."""

from __future__ import annotations

import importlib
import platform
import sys

import torch

from temporallens.utils.device import get_device

PACKAGES = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "einops": "einops",
    "gradio": "gradio",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pydantic": "pydantic",
    "PyYAML": "yaml",
    "rich": "rich",
    "safetensors": "safetensors",
    "scikit-learn": "sklearn",
    "SciPy": "scipy",
    "sentencepiece": "sentencepiece",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "typer": "typer",
    "wandb": "wandb",
}


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "installed"))


def main() -> int:
    expected_python = (3, 11)
    actual_python = sys.version_info[:2]
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    if actual_python != expected_python:
        print(f"ERROR: expected Python {expected_python[0]}.{expected_python[1]}")
        return 1

    failures: list[str] = []
    for display_name, module_name in PACKAGES.items():
        try:
            print(f"{display_name}: {version_of(module_name)}")
        except Exception as exc:  # report all failed imports in one pass
            failures.append(f"{display_name}: {exc}")

    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Selected device: {get_device()}")

    if failures:
        print("\nFailed imports:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

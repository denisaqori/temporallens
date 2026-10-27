"""Strict YAML loading: a repeated key is an error, never last-write-wins."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from temporallens.utils.yaml_io import DuplicateKeyError, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_a_repeated_key_is_refused_with_its_line() -> None:
    """PyYAML would keep the second value silently; a reader of the first would be misled."""
    text = "training:\n  epochs: 2\n  batch_size: 8\n  epochs: 500\n"
    assert yaml.safe_load(text)["training"]["epochs"] == 500, "premise: plain YAML keeps the last"
    with pytest.raises(DuplicateKeyError, match=r"duplicate YAML mapping key: 'epochs' \(line 4\)"):
        load_yaml(text)


def test_a_repeated_top_level_key_is_refused() -> None:
    with pytest.raises(DuplicateKeyError, match="duplicate YAML mapping key: 'model'"):
        load_yaml("model: {type: cnn1d}\nmodel: {type: other}\n")


def test_an_unhashable_key_is_refused() -> None:
    with pytest.raises(DuplicateKeyError, match="unhashable YAML mapping key"):
        load_yaml("? [a, b]\n: 1\n")


@pytest.mark.parametrize(
    "path",
    sorted(str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.glob("configs/**/*.yaml")),
)
def test_every_real_config_is_unambiguous_and_reads_the_same(path: str) -> None:
    """The strict loader changes nothing about a config that has no repeated keys."""
    text = (REPO_ROOT / path).read_text()
    assert load_yaml(text) == yaml.safe_load(text)

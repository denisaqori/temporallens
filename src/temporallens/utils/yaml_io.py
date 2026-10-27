"""Strict YAML loading for every file whose contents decide what a run does.

PyYAML's safe loader resolves a repeated mapping key by keeping the *last* value, silently. In a
config that means ``epochs: 2`` near the top and ``epochs: 500`` further down trains for 500 epochs
while a reader of the top sees 2, and nothing records that two values existed. The split manifest
already refused repeated keys; this module is that rule, shared, so configs and the manifest are
read the same way.

YAML merge keys (``<<:``) are not supported: they are resolved by the default mapping constructor
this module replaces, and no project file uses them.
"""

from __future__ import annotations

from typing import Any

import yaml
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver


class DuplicateKeyError(ValueError):
    """A YAML mapping repeats a key (or uses an unhashable one), so its meaning is ambiguous."""


class UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses repeated and unhashable mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        line = key_node.start_mark.line + 1
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise DuplicateKeyError(f"unhashable YAML mapping key: {key!r} (line {line})") from exc
        if duplicate:
            raise DuplicateKeyError(f"duplicate YAML mapping key: {key!r} (line {line})")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_yaml(text: str) -> Any:
    """Parse YAML text, refusing repeated mapping keys rather than keeping the last one."""
    return yaml.load(text, Loader=UniqueKeyLoader)

"""Validation helpers for robustness-target plan resolution."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def pending_field_paths(value: object, prefix: str = "") -> list[str]:
    """Return paths whose values are explicit ``pending_*`` sentinels."""
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(pending_field_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            paths.extend(pending_field_paths(child, child_prefix))
        return paths
    if isinstance(value, str) and value.startswith("pending_"):
        return [prefix]
    return []


def duplicate_target_names(targets: list[dict[str, Any]]) -> list[str]:
    """Return duplicate registry names in stable sorted order."""
    counts = Counter(target.get("name") for target in targets)
    return sorted(str(name) for name, count in counts.items() if count > 1)


def invalid_shared_k0_aliases(targets: list[dict[str, Any]]) -> dict[str, str]:
    """Return ``target -> missing alias`` entries from the robustness registry."""
    names = {target.get("name") for target in targets}
    return {
        str(target.get("name")): str(target["shared_k0_with"])
        for target in targets
        if "shared_k0_with" in target and target["shared_k0_with"] not in names
    }


def source_protocol_blockers(target: dict[str, Any]) -> list[str]:
    """Return blocker paths when a target's source experiment is not protocol-ready."""
    source = target.get("source_config")
    if source is None:
        return []

    source_path = Path(str(source))
    config = yaml.safe_load(source_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"source config {source_path} must contain a YAML mapping")

    protocol = config.get("protocol")
    if protocol is None:
        return []
    if not isinstance(protocol, dict):
        raise ValueError(f"source config {source_path} protocol must be a mapping")

    status = protocol.get("status")
    if status in (None, "ready"):
        return []

    blockers = [f"source_config.protocol.status={status}"]
    blocked_on = protocol.get("blocked_on", [])
    if not isinstance(blocked_on, list):
        raise ValueError(f"source config {source_path} protocol.blocked_on must be a list")
    blockers.extend(f"source_config.{path}" for path in blocked_on)
    return blockers

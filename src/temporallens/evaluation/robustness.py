"""Validation helpers for robustness-target plan resolution."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from temporallens.utils.yaml_io import load_yaml


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


def protocol_blockers(
    config: dict[str, Any],
    *,
    prefix: str = "",
    origin: str = "config",
    require_declared: bool = True,
) -> list[str]:
    """Return blocker paths when a loaded config's own protocol block is not ready.

    Shared by both gates so they cannot drift: a perturbation config checks itself, and a
    registry target checks the experiment that produced it. ``prefix`` is what the reported
    paths are relative to, so a caller can tell its own unresolved field from a target's.

    Readiness must be *declared*. A missing block or a missing status is a blocker, because
    silence is indistinguishable from someone having written ``ready``; pass
    ``require_declared=False`` only where a config is never reportable. ``status: ready`` with a
    non-empty ``blocked_on`` is a contradiction and raises: one of the two was not updated.
    """
    protocol = config.get("protocol")
    if protocol is None:
        return [f"{prefix}protocol=undeclared"] if require_declared else []
    if not isinstance(protocol, dict):
        raise ValueError(f"{origin} protocol must be a mapping")

    blocked_on = protocol.get("blocked_on", [])
    if not isinstance(blocked_on, list):
        raise ValueError(f"{origin} protocol.blocked_on must be a list")
    status = protocol.get("status")
    if status is None:
        return [f"{prefix}protocol.status=undeclared"]
    if not isinstance(status, str):
        raise ValueError(f"{origin} protocol.status must be a string, got {status!r}")
    if status == "ready":
        if blocked_on:
            raise ValueError(
                f"{origin} protocol.status is ready but blocked_on still lists {blocked_on}; "
                "resolve those fields or set a blocking status"
            )
        return []

    blockers = [f"{prefix}protocol.status={status}"]
    blockers.extend(f"{prefix}{path}" for path in blocked_on)
    return blockers


def source_protocol_blockers(target: dict[str, Any]) -> list[str]:
    """Return blocker paths when a target's source experiment is not protocol-ready."""
    source = target.get("source_config")
    if source is None:
        return ["source_config=undeclared"]

    source_path = Path(str(source))
    config = load_yaml(source_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"source config {source_path} must contain a YAML mapping")

    return protocol_blockers(config, prefix="source_config.", origin=f"source config {source_path}")

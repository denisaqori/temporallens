"""D43's keyed randomization, independent of global RNG state and execution order.

Protocol assignments use explicit NumPy PCG64 generators. Training retains its native
backend and existing seed contracts; this module never seeds Python, NumPy or PyTorch
globally. Persist realized assignments beside the provenance returned here: a seed does
not promise cross-version compatibility for higher-level distribution/permutation calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

CONTRACT_VERSION = "temporallens_rng_v1"
ASSIGNMENT_PRNG = "numpy_pcg64"


def _name(value: str, field: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        raise ValueError(f"{field} must be a lowercase ASCII identifier")


def _key(root_seed: int, scope: str, purpose: str, indices: Mapping[str, int]) -> dict[str, Any]:
    if type(root_seed) is not int or not 0 <= root_seed < 2**64:
        raise ValueError("root_seed must be an unsigned 64-bit Python integer")
    _name(scope, "scope")
    _name(purpose, "purpose")
    if not isinstance(indices, Mapping):
        raise ValueError("indices must map identifier names to nonnegative Python integers")
    for name, value in indices.items():
        _name(name, "index name")
        if type(value) is not int or value < 0:
            raise ValueError(f"index {name!r} must be a nonnegative Python integer")
    return {
        "contract": CONTRACT_VERSION,
        "root_seed": root_seed,
        "scope": scope,
        "purpose": purpose,
        "indices": dict(sorted(indices.items())),
    }


def _seed(key: dict[str, Any]) -> int:
    encoded = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(encoded.encode("utf-8")).digest()[:8], "big")


def derive_seed(root_seed: int, scope: str, purpose: str, indices: Mapping[str, int]) -> int:
    """Derive D43's unsigned 64-bit seed from a versioned, canonical JSON key.

    The key contains contract, root_seed, scope, purpose and named integer indices;
    serialization is sorted, compact ASCII JSON encoded as UTF-8. Interpret the first
    eight SHA-256 digest bytes as unsigned big-endian. Semantic stream identities belong
    to the experiment protocol: never add a strategy/subject/schedule ID to an object
    that must be shared across that axis. Generated repeat indices start at zero;
    existing dataset identifiers retain their original numbering.
    """
    return _seed(_key(root_seed, scope, purpose, indices))


def assignment_rng(
    root_seed: int, scope: str, purpose: str, indices: Mapping[str, int]
) -> np.random.Generator:
    """Return a fresh local PCG64 generator for one declared random assignment.

    Reconstructing the same key starts the same stream; drawing from a different key
    cannot advance it. Callers must define input ordering, allocation/permutation calls
    and assignment persistence separately before a protocol is executable.
    """
    return np.random.Generator(np.random.PCG64(derive_seed(root_seed, scope, purpose, indices)))


def randomization_provenance(
    root_seed: int, scope: str, purpose: str, indices: Mapping[str, int]
) -> dict[str, Any]:
    """Describe an assignment stream without drawing from it or touching global state.

    This is seed provenance, not the assignment artifact itself. Runners must also save
    realized stable IDs in order, their source-pool identity and assignment SHA-256.
    High-volume training randomness instead records component seeds/backend/draw IDs;
    restoring a mid-training stream additionally requires checkpoint RNG state.
    """
    key = _key(root_seed, scope, purpose, indices)
    return {
        "key": key,
        "derived_seed": _seed(key),
        "assignment_prng": ASSIGNMENT_PRNG,
        "numpy_version": np.__version__,
    }

"""D43's stream identity must not depend on call order, globals or dict order."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pytest
import torch

from temporallens.utils.randomness import (
    ASSIGNMENT_PRNG,
    CONTRACT_VERSION,
    assignment_rng,
    derive_seed,
    randomization_provenance,
)


def test_seed_serialization_is_exact_and_index_order_independent() -> None:
    expected_json = (
        '{"contract":"temporallens_rng_v1","indices":{"schedule_index":0,"subject_id":5},'
        '"purpose":"acquisition_schedule","root_seed":42,"scope":"g3_g4_personalization"}'
    )
    expected_seed = int.from_bytes(hashlib.sha256(expected_json.encode()).digest()[:8], "big")
    for indices in (
        {"subject_id": 5, "schedule_index": 0},
        {"schedule_index": 0, "subject_id": 5},
    ):
        assert (
            derive_seed(42, "g3_g4_personalization", "acquisition_schedule", indices)
            == expected_seed
        )


def test_generators_are_fresh_keyed_streams_not_a_shared_mutable_rng() -> None:
    kwargs = {"root_seed": 42, "scope": "f2", "purpose": "outer_assignment", "indices": {}}
    first = assignment_rng(**kwargs).permutation(100)
    assignment_rng(42, "f2", "inner_assignment", {}).random(1000)
    assert np.array_equal(first, assignment_rng(**kwargs).permutation(100))
    assert isinstance(assignment_rng(**kwargs).bit_generator, np.random.PCG64)
    assert derive_seed(42, "f2", "outer_assignment", {}) != derive_seed(
        42, "f2", "inner_assignment", {}
    )


def test_helper_does_not_seed_or_consume_global_random_state() -> None:
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    torch_before = torch.get_rng_state().clone()
    assignment_rng(42, "diagnostic", "assignment", {}).permutation(20)
    randomization_provenance(42, "diagnostic", "assignment", {})
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert python_before == random.getstate()


def test_provenance_records_exact_key_seed_backend_and_version() -> None:
    indices = {"subject_id": 5, "schedule_index": 0}
    record = randomization_provenance(42, "g3_g4_personalization", "acquisition_schedule", indices)
    assert record == {
        "key": {
            "contract": CONTRACT_VERSION,
            "root_seed": 42,
            "scope": "g3_g4_personalization",
            "purpose": "acquisition_schedule",
            "indices": {"subject_id": 5, "schedule_index": 0},
        },
        "derived_seed": derive_seed(42, "g3_g4_personalization", "acquisition_schedule", indices),
        "assignment_prng": "numpy_pcg64",
        "numpy_version": np.__version__,
    }
    indices["schedule_index"] = 99
    assert record["key"]["indices"]["schedule_index"] == 0
    assert list(record["key"]["indices"]) == ["schedule_index", "subject_id"]
    assert ASSIGNMENT_PRNG == "numpy_pcg64"
    json.dumps(record, allow_nan=False)


def test_semantic_scope_purpose_and_index_changes_get_distinct_seeds() -> None:
    seeds = {
        derive_seed(42, "scope", "purpose", {"seed_index": 0}),
        derive_seed(43, "scope", "purpose", {"seed_index": 0}),
        derive_seed(42, "other_scope", "purpose", {"seed_index": 0}),
        derive_seed(42, "scope", "other_purpose", {"seed_index": 0}),
        derive_seed(42, "scope", "purpose", {"seed_index": 1}),
    }
    assert len(seeds) == 5


def test_invalid_index_mapping_and_names_are_refused() -> None:
    with pytest.raises(ValueError, match="indices"):
        derive_seed(42, "scope", "purpose", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index name"):
        derive_seed(42, "scope", "purpose", {"Seed": 0})


@pytest.mark.parametrize("root_seed", [-1, 2**64, True, 42.0, "42"])
def test_invalid_root_seed_is_refused(root_seed: object) -> None:
    with pytest.raises(ValueError, match="root_seed"):
        derive_seed(root_seed, "scope", "purpose", {})  # type: ignore[arg-type]


@pytest.mark.parametrize("root_seed", [0, 2**64 - 1])
def test_valid_root_seed_boundaries_are_supported(root_seed: int) -> None:
    seed = derive_seed(root_seed, "scope", "purpose", {})
    assert type(seed) is int and 0 <= seed < 2**64


@pytest.mark.parametrize("name", ["", "has space", "G3", "a:b", "é", 4])
def test_ambiguous_or_noncanonical_names_are_refused(name: object) -> None:
    with pytest.raises(ValueError, match="scope"):
        derive_seed(42, name, "purpose", {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="purpose"):
        derive_seed(42, "scope", name, {})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, True, 1.0, "1", None])
def test_invalid_indices_are_refused(value: object) -> None:
    with pytest.raises(ValueError, match="index"):
        derive_seed(42, "scope", "purpose", {"seed_index": value})  # type: ignore[dict-item]

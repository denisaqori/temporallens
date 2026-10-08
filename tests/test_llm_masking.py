"""Padding-side-independent mask behavior for future frozen-LLM adapters."""

from __future__ import annotations

import pytest
import torch

from temporallens.models.llm.masking import (
    mask_derived_position_ids,
    prepend_soft_prefix_mask,
    rightmost_nonpadding_indices,
)


@pytest.mark.parametrize(
    ("mask", "expected"),
    (
        ([0, 0, 1, 1], 3),
        ([1, 1, 0, 0], 1),
        ([1, 0, 1, 0], 2),
        ([1, 1, 1, 1], 3),
    ),
)
def test_rightmost_nonpadding_index_handles_padding_and_interior_zeros(
    mask: list[int], expected: int
) -> None:
    assert rightmost_nonpadding_indices(torch.tensor([mask])) == expected


def test_boolean_masks_are_cast_before_argmax() -> None:
    mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    assert rightmost_nonpadding_indices(mask) == 2


@pytest.mark.parametrize("dtype", (torch.int64, torch.bool))
def test_soft_prefix_combined_mask_drives_positions_and_pooling(dtype: torch.dtype) -> None:
    text_mask = torch.tensor(
        [
            [0, 0, 1, 1],  # left padded
            [1, 1, 0, 0],  # right padded
        ],
        dtype=dtype,
    )
    combined = prepend_soft_prefix_mask(text_mask, num_soft_tokens=2)

    torch.testing.assert_close(
        combined.to(torch.int64),
        torch.tensor(
            [
                [1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ]
        ),
    )
    torch.testing.assert_close(
        mask_derived_position_ids(combined),
        torch.tensor(
            [
                [0, 1, 0, 0, 2, 3],
                [0, 1, 2, 3, 0, 0],
            ]
        ),
    )
    torch.testing.assert_close(rightmost_nonpadding_indices(combined), torch.tensor([5, 3]))


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor([[0, 0]]),
        torch.tensor([[0.0, 0.5, 1.0]]),
        torch.ones((1, 2, 2), dtype=torch.int64),
        torch.empty((1, 0), dtype=torch.int64),
    ),
)
def test_invalid_attention_masks_are_rejected(mask: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        rightmost_nonpadding_indices(mask)


def test_token_all_pad_is_rejected_before_prefix_is_added() -> None:
    with pytest.raises(ValueError, match="non-padding token"):
        prepend_soft_prefix_mask(torch.zeros((1, 4), dtype=torch.bool), num_soft_tokens=2)


def test_soft_prefix_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        prepend_soft_prefix_mask(torch.ones((1, 2), dtype=torch.int64), num_soft_tokens=0)

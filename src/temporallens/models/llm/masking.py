"""Attention-mask helpers shared by frozen language-model input paths."""

from __future__ import annotations

import torch
from torch import Tensor


def _validated_binary_mask(attention_mask: Tensor) -> Tensor:
    """Return a 2-D mask as integers, rejecting ambiguous or empty rows."""
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape (batch, sequence), got {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[1] == 0:
        raise ValueError("attention_mask must contain at least one sequence position")

    mask = attention_mask.to(dtype=torch.int64)
    if not torch.equal(mask, attention_mask):
        raise ValueError("attention_mask must contain only binary 0/1 values")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must contain only binary 0/1 values")
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every attention-mask row must contain at least one non-padding token")
    return mask


def prepend_soft_prefix_mask(text_attention_mask: Tensor, num_soft_tokens: int) -> Tensor:
    """Prepend one unmasked position for every continuous soft-prefix embedding."""
    _validated_binary_mask(text_attention_mask)
    if num_soft_tokens <= 0:
        raise ValueError(f"num_soft_tokens must be positive, got {num_soft_tokens}")
    prefix = torch.ones(
        (text_attention_mask.shape[0], num_soft_tokens),
        dtype=text_attention_mask.dtype,
        device=text_attention_mask.device,
    )
    return torch.cat((prefix, text_attention_mask), dim=-1)


def mask_derived_position_ids(attention_mask: Tensor) -> Tensor:
    """Assign positions by cumulative non-padding count, with padding positions set to zero."""
    mask = _validated_binary_mask(attention_mask)
    position_ids = mask.cumsum(dim=-1) - 1
    return position_ids.masked_fill(mask == 0, 0)


def rightmost_nonpadding_indices(attention_mask: Tensor) -> Tensor:
    """Return the rightmost mask-1 index for each row, independent of padding side."""
    mask = _validated_binary_mask(attention_mask)
    sequence_length = mask.shape[-1]
    indices = sequence_length - 1 - mask.flip(dims=(-1,)).argmax(dim=-1)
    rows = torch.arange(mask.shape[0], device=mask.device)
    if not torch.all(mask[rows, indices] == 1):  # defensive check on the selected positions
        raise AssertionError("rightmost non-padding index selected a masked position")
    return indices

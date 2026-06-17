"""Soft mixture utilities for combining multiple chart/slot adapter outputs."""

from typing import List, Optional

from torch import Tensor


def mix_chart_outputs(
    deltas: List[Tensor],
    weights: Tensor,
) -> Tensor:
    """
    Soft-mixture of chart-adapter residuals.

    delta_mixed = sum_c weight_c * delta_c

    Args:
        deltas: list of residual tensors, each [B, D].
        weights: mixture weights of shape [B, num_charts].

    Returns:
        Mixed residual of shape [B, D].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def mix_slot_outputs(
    deltas: List[Tensor],
    weights: Tensor,
) -> Tensor:
    """
    Soft-mixture of slot-adapter residuals.

    delta_mixed = sum_s weight_s * delta_s

    Args:
        deltas: list of residual tensors per slot, each [B, D].
        weights: mixture weights of shape [B, num_slots].

    Returns:
        Mixed residual of shape [B, D].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def normalize_topk_probs(
    probs: Tensor,
    topk_indices: Tensor,
) -> Tensor:
    """
    Renormalize probabilities after top-k selection.

    Args:
        probs: original probabilities [B, K].
        topk_indices: selected indices [B, k].

    Returns:
        Renormalized probabilities of shape [B, k], summing to 1.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def apply_topk_mask(
    values: Tensor,
    topk_indices: Tensor,
    fill_value: float = 0.0,
) -> Tensor:
    """
    Apply top-k mask to values tensor.

    Args:
        values: input tensor [B, K, ...].
        topk_indices: selected indices [B, k].
        fill_value: value for non-selected entries.

    Returns:
        Masked tensor of same shape as values.
    """
    raise NotImplementedError("Phase-0 skeleton only.")

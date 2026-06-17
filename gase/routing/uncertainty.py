"""Uncertainty estimation for routing decisions."""

from typing import List

from torch import Tensor

from ..atlas.chart_state import ChartState
from ..slots.slot_state import SlotState


def compute_entropy(probs: Tensor) -> Tensor:
    """
    Compute per-sample entropy of a probability distribution.

    H_i = -sum_k p_{i,k} * log(p_{i,k})

    Args:
        probs: probability distribution of shape [B, K].

    Returns:
        Entropy values of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_top_margin(probs: Tensor) -> Tensor:
    """
    Compute margin between top-1 and top-2 probabilities.

    margin_i = max_k p_{i,k} - second_max_k p_{i,k}

    Args:
        probs: probability distribution of shape [B, K].

    Returns:
        Margin values of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def is_uncertain(
    probs: Tensor,
    entropy_threshold: float,
    margin_threshold: float,
) -> Tensor:
    """
    Detect uncertain samples via entropy and margin criteria.

    A sample is uncertain if:
        entropy > entropy_threshold OR margin < margin_threshold

    Args:
        probs: probabilities [B, K].
        entropy_threshold: max allowed entropy.
        margin_threshold: min required margin.

    Returns:
        Boolean mask of shape [B], True for uncertain samples.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_chart_uncertainty(
    h_chart: Tensor,
    chart_states: List[ChartState],
) -> Tensor:
    """
    Compute per-sample chart assignment uncertainty.

    Args:
        h_chart: features [B, D].
        chart_states: available charts.

    Returns:
        Uncertainty scores of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_slot_uncertainty(
    h_chart: Tensor,
    chart_state: ChartState,
    slot_states: List[SlotState],
) -> Tensor:
    """
    Compute per-sample slot assignment uncertainty within a chart.

    Args:
        h_chart: features [B, D].
        chart_state: parent chart.
        slot_states: available slots in this chart.

    Returns:
        Uncertainty scores of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")

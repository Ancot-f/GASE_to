"""Posterior computation for chart assignment and uncertainty detection."""

from typing import List, Tuple

from torch import Tensor

from .chart_state import ChartState


def compute_chart_nll(h_chart: Tensor, chart_state: ChartState) -> Tensor:
    """
    Compute negative log-likelihood of features under a PPCA chart model.

    NLL(h | c) = -log p(h | chart_c)

    Args:
        h_chart: features of shape [B, D].
        chart_state: chart with fitted PPCA parameters (mu, U, eigvals, sigma_perp).

    Returns:
        NLL values of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_chart_posterior(
    h_chart: Tensor,
    chart_states: List[ChartState],
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute posterior probability p(chart | h) for each chart.

    Uses Bayes rule with PPCA likelihood and chart priors,
    optionally tempered.

    Args:
        h_chart: features of shape [B, D].
        chart_states: list of chart states.
        temperature: softmax temperature for sharpening/flattening.

    Returns:
        Posterior probabilities of shape [B, num_charts].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_chart_entropy(chart_probs: Tensor) -> Tensor:
    """
    Compute per-sample entropy of chart posterior.

    H_i = -sum_c p(c|h_i) * log p(c|h_i)

    Args:
        chart_probs: posterior of shape [B, num_charts].

    Returns:
        Entropy values of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def select_top_m_charts(
    chart_probs: Tensor,
    top_m: int,
) -> Tuple[Tensor, Tensor]:
    """
    Select top-m charts per sample and renormalize.

    Args:
        chart_probs: posterior of shape [B, num_charts].
        top_m: number of top charts to keep.

    Returns:
        Tuple of (top_m_probs [B, top_m], top_m_indices [B, top_m]).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def detect_boundary_samples(
    chart_probs: Tensor,
    margin: float,
) -> Tensor:
    """
    Detect samples near chart boundaries (ambiguous assignment).

    A sample is on the boundary if the top two chart probabilities
    differ by less than margin.

    Args:
        chart_probs: posterior of shape [B, num_charts].
        margin: probability margin threshold.

    Returns:
        Boolean mask of shape [B], True for boundary samples.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def detect_uncovered_samples(
    chart_probs: Tensor,
    threshold: float,
) -> Tensor:
    """
    Detect samples not well-covered by any chart.

    A sample is uncovered if max_c p(c|h) < threshold.

    Args:
        chart_probs: posterior of shape [B, num_charts].
        threshold: minimum probability to be considered covered.

    Returns:
        Boolean mask of shape [B], True for uncovered samples.
    """
    raise NotImplementedError("Phase-0 skeleton only.")

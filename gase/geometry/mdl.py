"""Minimum Description Length (MDL) for chart acceptance decisions."""

from typing import List

from torch import Tensor

from ..atlas.chart_state import ChartState


def compute_chart_complexity(
    chart_state: ChartState,
    n_samples: int,
) -> float:
    """
    Compute the description length (complexity) of a chart.

    Cost includes: mean vector, basis matrix, eigenvalues, sigma_perp,
    and per-sample latent codes.

    Args:
        chart_state: chart to evaluate.
        n_samples: number of samples assigned to chart.

    Returns:
        Chart complexity in nats.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_nll_gain(
    chart_state: ChartState,
    h_assigned: Tensor,
) -> float:
    """
    Compute the NLL improvement from modeling with chart vs. null model.

    NLL_gain = NLL_null - NLL_chart

    Args:
        chart_state: chart to evaluate.
        h_assigned: features assigned to chart [N, D].

    Returns:
        NLL gain (positive = better fit).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_mdl_gain(
    chart_state: ChartState,
    h_assigned: Tensor,
    lambda_param: float = 1.0,
) -> float:
    """
    Compute MDL gain: NLL improvement minus complexity penalty.

    MDL_gain = NLL_gain - lambda * complexity

    Positive MDL gain means the chart is worth keeping.

    Args:
        chart_state: chart to evaluate.
        h_assigned: features assigned to chart [N, D].
        lambda_param: complexity penalty weight.

    Returns:
        MDL gain (positive = accept chart).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def should_accept_new_chart(
    candidate: ChartState,
    h_assigned: Tensor,
    existing_charts: List[ChartState],
    lambda_param: float = 1.0,
) -> bool:
    """
    MDL-based decision for accepting a new chart.

    Accept if the MDL gain of adding the new chart is positive,
    considering both the new chart's gain and the effect on
    existing charts' NLL.

    Args:
        candidate: proposed new chart.
        h_assigned: features assigned to candidate [N, D].
        existing_charts: current charts in layer.
        lambda_param: complexity penalty.

    Returns:
        True if chart should be accepted.
    """
    raise NotImplementedError("Phase-0 skeleton only.")

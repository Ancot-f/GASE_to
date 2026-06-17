"""Chart quality and diagnostic metrics."""

from typing import List

from torch import Tensor

from .chart_state import ChartState


def compute_chart_compactness(
    chart_state: ChartState,
    h_assigned: Tensor,
) -> float:
    """
    Compute compactness of a chart: average NLL or reconstruction error.

    Args:
        chart_state: chart to evaluate.
        h_assigned: features assigned to this chart [N, D].

    Returns:
        Compactness score (lower is more compact).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_chart_support(
    chart_state: ChartState,
    h_chart: Tensor,
    threshold: float = 0.75,
) -> int:
    """
    Count samples within the chart's inlier radius.

    Args:
        chart_state: chart to evaluate.
        h_chart: features to test [N, D].
        threshold: posterior threshold for assignment.

    Returns:
        Number of inlier samples.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_tangent_stability(
    chart_state: ChartState,
    h_assigned: Tensor,
) -> float:
    """
    Compute tangent space stability: how well the PPCA subspace
    explains the variance of assigned features.

    Args:
        chart_state: chart to evaluate.
        h_assigned: features [N, D].

    Returns:
        Tangent space stability score in [0, 1].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_normal_residual(
    chart_state: ChartState,
    h_assigned: Tensor,
) -> Tensor:
    """
    Compute normal-space residual: component orthogonal to chart basis.

    r_normal = (I - U U^T) (h - mu)

    Args:
        chart_state: chart with basis U.
        h_assigned: features [N, D].

    Returns:
        Normal residuals of shape [N, D].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_chart_overlap(
    chart_a: ChartState,
    chart_b: ChartState,
) -> float:
    """
    Compute overlap between two charts.

    Uses subspace similarity and Mahalanobis-based boundary overlap.

    Args:
        chart_a: first chart.
        chart_b: second chart.

    Returns:
        Overlap score in [0, 1], higher = more overlap.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_chart_quality(
    chart_state: ChartState,
    h_assigned: Tensor,
    delta_teacher: Tensor,
) -> dict:
    """
    Compute comprehensive chart quality metrics.

    Args:
        chart_state: chart to evaluate.
        h_assigned: features [N, D].
        delta_teacher: teacher residuals [N, D] for fit evaluation.

    Returns:
        Dict with keys: compactness, stability, residual_fit_r2, n_support, etc.
    """
    raise NotImplementedError("Phase-0 skeleton only.")

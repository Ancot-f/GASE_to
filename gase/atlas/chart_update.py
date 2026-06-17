"""Chart update functions: EMA-based and Grassmann-based geometry updates."""

from typing import Optional

from torch import Tensor

from .chart_state import ChartState


def update_chart_statistics(
    chart_state: ChartState,
    h_chart: Tensor,
    assignment_weights: Tensor,
    momentum: float = 0.9,
) -> ChartState:
    """
    Update chart mu, basis, and eigenvalues via weighted EMA.

    This is the main entry point for online chart updates.

    Args:
        chart_state: current chart state to update.
        h_chart: features assigned to this chart [N, D].
        assignment_weights: soft assignment weights [N] in [0, 1].
        momentum: EMA decay factor.

    Returns:
        Updated ChartState.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def ema_update_mu(
    mu_old: Tensor,
    h_chart: Tensor,
    weights: Tensor,
    momentum: float,
) -> Tensor:
    """
    EMA update chart mean.

    mu_new = momentum * mu_old + (1 - momentum) * weighted_mean(h)

    Args:
        mu_old: current mean [D].
        h_chart: features [N, D].
        weights: assignment weights [N].
        momentum: EMA decay.

    Returns:
        Updated mean [D].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def ema_update_eigvals(
    eigvals_old: Tensor,
    h_chart: Tensor,
    mu_new: Tensor,
    U: Tensor,
    weights: Tensor,
    momentum: float,
) -> Tensor:
    """
    EMA update chart eigenvalues.

    Args:
        eigvals_old: current eigenvalues [rank].
        h_chart: features [N, D].
        mu_new: updated mean [D].
        U: current basis [D, rank].
        weights: assignment weights [N].
        momentum: EMA decay.

    Returns:
        Updated eigenvalues [rank].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def grassmann_update_basis(
    U_old: Tensor,
    h_chart: Tensor,
    mu_new: Tensor,
    weights: Tensor,
    momentum: float,
) -> Tensor:
    """
    Update chart basis via Grassmann geodesic interpolation.

    Uses principal angles and geodesic averaging on the Grassmann
    manifold to smoothly update the subspace basis.

    Args:
        U_old: current basis [D, rank].
        h_chart: features [N, D].
        mu_new: updated mean [D].
        weights: assignment weights [N].
        momentum: interpolation factor in [0, 1].

    Returns:
        Updated basis [D, rank].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def update_chart_quality(
    chart_state: ChartState,
    h_chart: Tensor,
    delta_teacher: Optional[Tensor] = None,
) -> ChartState:
    """
    Recompute chart quality metrics (compactness, stability, etc.).

    Args:
        chart_state: chart to evaluate.
        h_chart: features assigned to chart [N, D].
        delta_teacher: optional teacher residuals [N, D] for fit evaluation.

    Returns:
        Updated ChartState with refreshed quality dict.
    """
    raise NotImplementedError("Phase-0 skeleton only.")

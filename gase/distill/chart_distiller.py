"""ChartAdapterDistiller: distills task-adapter residuals into chart-adapters."""

from typing import Dict, List, Tuple

from torch import Tensor

from ..atlas.chart_state import ChartState
from ..slots.slot_state import SlotState


class ChartAdapterDistiller:
    """
    Distills task-adapter teacher residuals into chart-slot adapters.

    For each (chart, slot) pair, learns the chart-adapter parameters
    (P, R, B, b or MLP weights) to minimize the discrepancy between
    the chart-adapter residual and the teacher residual.

    The distillation is local: each chart-adapter learns only from
    samples assigned to that chart-slot pair.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: distillation config dict.
        """
        self.config = config

    def distill_for_layer(
        self,
        layer_id: int,
        layer_cache,
        chart_states: List[ChartState],
        slot_states: Dict[int, List[SlotState]],
    ) -> None:
        """
        Distill all chart-adapters for a single layer.

        Args:
            layer_id: ViT block index.
            layer_cache: collected LayerFeatureBatch list.
            chart_states: all charts in this layer.
            slot_states: dict mapping chart_id -> list of SlotState.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def distill_for_chart_slot(
        self,
        chart_state: ChartState,
        slot_state: SlotState,
        h_chart: Tensor,
        delta_teacher: Tensor,
    ) -> None:
        """
        Distill a single chart-slot adapter.

        Args:
            chart_state: parent chart.
            slot_state: slot to distill for.
            h_chart: features assigned to this (chart, slot) [N, D].
            delta_teacher: teacher residuals [N, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def fit_linear_chart_adapter(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        P: Tensor,
        R: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Fit linear chart-adapter (B, b) via least squares.

        min_{B,b} ||delta_teacher - (b + R @ B @ P^T @ (h - mu))||^2

        Args:
            h_chart: features [N, D].
            delta_teacher: teacher residuals [N, D].
            P: input projection basis [D, input_rank].
            R: output projection basis [output_rank, D].

        Returns:
            Tuple of (B [output_rank, input_rank], b [D]).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def fit_mlp_chart_adapter(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        P: Tensor,
        R: Tensor,
    ) -> None:
        """
        Fit MLP chart-adapter via gradient descent.

        Args:
            h_chart: features [N, D].
            delta_teacher: teacher residuals [N, D].
            P: input projection basis.
            R: output projection basis.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_projection_bases(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        input_rank: int,
        output_rank: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute P and R bases from h_chart and delta_teacher.

        P: top input_rank right singular vectors of cross-covariance.
        R: top output_rank left singular vectors of cross-covariance.

        Args:
            h_chart: features [N, D].
            delta_teacher: teacher residuals [N, D].
            input_rank: rank of P.
            output_rank: rank of R.

        Returns:
            Tuple of (P [D, input_rank], R [output_rank, D]).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_distill_losses(
        self,
        delta_chart: Tensor,
        delta_teacher: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute all distillation losses for chart-adapter training.

        Args:
            delta_chart: chart-adapter residuals [B, D].
            delta_teacher: teacher residuals [B, D].

        Returns:
            Dict of loss_name -> scalar loss value.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

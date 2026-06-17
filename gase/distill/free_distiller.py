"""FreeAdapterDistiller: learns free-adapter to absorb residual leftover."""

from typing import Dict, List, Optional

from torch import Tensor

from ..atlas.chart_state import ChartState
from ..slots.slot_state import SlotState


class FreeAdapterDistiller:
    """
    Trains the free-adapter to absorb residual leftover that
    chart/slot adapters cannot explain.

    The free target is:
        delta_free_target = delta_teacher - delta_chart
    for samples where chart-adapters provide poor fit.
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
        Train free-adapter for a single layer.

        Args:
            layer_id: ViT block index.
            layer_cache: collected LayerFeatureBatch list.
            chart_states: all charts in this layer.
            slot_states: dict mapping chart_id -> list of SlotState.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_free_targets(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        delta_chart: Tensor,
    ) -> Tensor:
        """
        Compute target residual for free-adapter.

        delta_free_target = delta_teacher - delta_chart

        Args:
            h_chart: features [B, D].
            delta_teacher: teacher residuals [B, D].
            delta_chart: chart-adapter residuals [B, D] (best available).

        Returns:
            Free-adapter target of shape [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def select_free_samples(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        delta_chart: Tensor,
    ) -> Tensor:
        """
        Select samples where free-adapter should be active.

        Samples with high chart-adapter fit error are candidates
        for free-adapter.

        Args:
            h_chart: features [B, D].
            delta_teacher: teacher residuals [B, D].
            delta_chart: chart-adapter residuals [B, D].

        Returns:
            Boolean mask of shape [B], True for selected samples.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def train_free_adapter(
        self,
        h_chart: Tensor,
        delta_free_target: Tensor,
    ) -> None:
        """
        Train the free-adapter on selected samples.

        Args:
            h_chart: features [N, D] for training.
            delta_free_target: target residuals [N, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

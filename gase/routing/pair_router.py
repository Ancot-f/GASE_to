"""ChartSlotPairRouter: joint chart+slot routing for inference."""

from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch import nn

from ..atlas.chart_state import ChartState
from ..slots.slot_state import SlotState


class ChartSlotPairRouter(nn.Module):
    """
    Joint chart-slot pair router.

    For inference, selects the best (chart, slot) pairs:
    1. Route to top-m charts (via ProbabilisticChartRouter).
    2. Within each selected chart, route to top-k slots.
    3. Combine chart and slot probabilities for weighted residual mixing.

    This produces a set of (chart_id, slot_id, weight) triples
    that determine how to combine chart-adapter outputs.
    """

    def __init__(
        self,
        dim: int,
        top_m: int = 2,
        top_k: int = 2,
        input_rank: int = 8,
        use_soft_routing: bool = True,
    ):
        """
        Args:
            dim: feature dimension D.
            top_m: number of top charts to select.
            top_k: number of top slots per chart.
            input_rank: rank of slot input projection (for key matching).
            use_soft_routing: if True, soft-mix residuals; if False, hard-select.
        """
        super().__init__()
        self.dim = dim
        self.top_m = top_m
        self.top_k = top_k
        self.input_rank = input_rank
        self.use_soft_routing = use_soft_routing

    def forward(
        self,
        h_chart: Tensor,
        chart_states: List[ChartState],
        slot_states_by_chart: Dict[int, List[SlotState]],
    ):
        """
        Route to (chart, slot) pairs and compute weights.

        Args:
            h_chart: features of shape [B, D].
            chart_states: available charts.
            slot_states_by_chart: dict chart_id -> list of SlotState.

        Returns:
            Routing information including pair weights and selected ids.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_pair_scores(
        self,
        h_chart: Tensor,
        chart_states: List[ChartState],
        slot_states_by_chart: Dict[int, List[SlotState]],
    ) -> Tensor:
        """
        Compute scores for all (chart, slot) pairs.

        pair_score = chart_prob * slot_prob_within_chart

        Args:
            h_chart: features [B, D].
            chart_states: available charts.
            slot_states_by_chart: slots per chart.

        Returns:
            Pair scores of shape [B, total_pairs].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_pair_probs(
        self,
        pair_scores: Tensor,
    ) -> Tensor:
        """
        Compute normalized pair probabilities.

        Args:
            pair_scores: raw scores [B, total_pairs].

        Returns:
            Pair probabilities of shape [B, total_pairs].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def select_top_pairs(
        self,
        pair_probs: Tensor,
        chart_states: List[ChartState],
        slot_states_by_chart: Dict[int, List[SlotState]],
    ):
        """
        Select top-k' pairs and return their (chart_id, slot_id, weight).

        Args:
            pair_probs: pair probabilities [B, total_pairs].
            chart_states: charts for index mapping.
            slot_states_by_chart: slots for index mapping.

        Returns:
            Tuple of (top_weights [B, K], chart_ids [B, K], slot_ids [B, K]).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

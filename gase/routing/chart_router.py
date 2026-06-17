"""ProbabilisticChartRouter: selects top-m charts via PPCA posterior."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from ..atlas.chart_state import ChartState


class ProbabilisticChartRouter(nn.Module):
    """
    Probabilistic chart router based on PPCA likelihood.

    For each input feature h_chart:
    1. Compute p(chart | h) via Bayes rule with PPCA models.
    2. Select top-m charts per sample.
    3. Compute a free-gate signal for uncovered samples.

    This is a non-parametric router: it uses the chart's PPCA
    parameters directly without additional learned weights.
    """

    def __init__(
        self,
        dim: int,
        top_m: int = 2,
        temperature: float = 1.0,
        use_entropy_gate: bool = True,
        entropy_threshold: float = 1.0,
    ):
        """
        Args:
            dim: feature dimension D.
            top_m: number of top charts to select.
            temperature: softmax temperature for posterior sharpening.
            use_entropy_gate: whether to compute a free-adapter gate from entropy.
            entropy_threshold: entropy above which free-adapter is triggered.
        """
        super().__init__()
        self.dim = dim
        self.top_m = top_m
        self.temperature = temperature
        self.use_entropy_gate = use_entropy_gate
        self.entropy_threshold = entropy_threshold

    def forward(
        self,
        h_chart: Tensor,
        chart_states: List[ChartState],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Route features to top-m charts.

        Args:
            h_chart: features of shape [B, D].
            chart_states: available charts in this layer.

        Returns:
            Tuple of:
                - top_probs: [B, top_m] renormalized posterior.
                - top_indices: [B, top_m] selected chart indices.
                - free_gate: [B] optional gate values in [0, 1].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_nll_scores(
        self,
        h_chart: Tensor,
        chart_states: List[ChartState],
    ) -> Tensor:
        """
        Compute negative log-likelihood scores for all charts.

        Args:
            h_chart: features [B, D].
            chart_states: available charts.

        Returns:
            NLL scores of shape [B, num_charts].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_chart_probs(
        self,
        h_chart: Tensor,
        chart_states: List[ChartState],
    ) -> Tensor:
        """
        Compute chart posterior probabilities.

        Args:
            h_chart: features [B, D].
            chart_states: available charts.

        Returns:
            Posterior of shape [B, num_charts].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def select_top_charts(
        self,
        chart_probs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Select top-m charts and renormalize.

        Args:
            chart_probs: posterior [B, num_charts].

        Returns:
            Tuple of (top_probs [B, top_m], top_indices [B, top_m]).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_free_gate(
        self,
        chart_probs: Tensor,
    ) -> Tensor:
        """
        Compute free-adapter gate from chart assignment uncertainty.

        gate_i = sigmoid((entropy_i - threshold) / temperature)

        Args:
            chart_probs: posterior [B, num_charts].

        Returns:
            Gate values of shape [B] in [0, 1].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

"""RoutingReporter: summarizes routing behavior and quality."""

from typing import Dict, List

from torch import Tensor

from ..atlas.chart_state import ChartState


class RoutingReporter:
    """
    Generates reports on routing behavior.

    Tracks chart usage distribution, slot usage entropy,
    routing confidence, and fallback rates.
    """

    def __init__(self, writer=None):
        """
        Args:
            writer: optional TensorBoard/MLflow writer.
        """
        self.writer = writer

    def summarize_chart_routing(
        self,
        chart_probs: Tensor,
        chart_states: List[ChartState],
    ) -> Dict:
        """
        Summarize chart routing statistics.

        Args:
            chart_probs: chart posterior [N, num_charts].
            chart_states: available charts.

        Returns:
            Dict with keys: mean_entropy, top1_rate, usage_distribution, etc.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def summarize_slot_routing(
        self,
        slot_probs: Tensor,
        chart_states: List[ChartState],
    ) -> Dict:
        """
        Summarize slot routing statistics.

        Args:
            slot_probs: slot probabilities per sample.
            chart_states: parent charts.

        Returns:
            Dict with slot usage metrics.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def summarize_pair_routing(
        self,
        pair_weights: Tensor,
        chart_ids: Tensor,
        slot_ids: Tensor,
    ) -> Dict:
        """
        Summarize (chart, slot) pair routing.

        Args:
            pair_weights: pair selection weights [N, K].
            chart_ids: selected chart ids [N, K].
            slot_ids: selected slot ids [N, K].

        Returns:
            Dict with pair usage statistics.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_route_fit_agreement(
        self,
        routing_scores: Tensor,
        fit_errors: Tensor,
    ) -> float:
        """
        Compute agreement between router scores and actual fit quality.

        High agreement means the router correctly predicts which
        adapters produce the best residuals.

        Args:
            routing_scores: router-assigned scores [N, K].
            fit_errors: actual fit errors [N, K].

        Returns:
            Agreement score in [0, 1].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def log_routing_entropy(
        self,
        chart_probs: Tensor,
        slot_probs: Tensor,
    ) -> None:
        """
        Log routing entropy histograms.

        Args:
            chart_probs: chart posterior [N, num_charts].
            slot_probs: slot probabilities [N, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

"""ChartMergeSplitManager: merge and split operations for chart maintenance."""

from typing import List, Optional, Tuple

from torch import Tensor

from .chart_state import ChartState


class ChartMergeSplitManager:
    """
    Manages chart merge and split operations.

    - Merging: combines two overlapping/drifting charts into one.
    - Splitting: divides a saturated or multimodal chart into two.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: merge/split configuration.
        """
        self.config = config

    def find_merge_candidates(
        self,
        chart_states: List[ChartState],
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of charts that are candidates for merging.

        Criteria: high subspace similarity, overlapping support,
        similar normal residual variance.

        Args:
            chart_states: all charts in a layer.
            similarity_threshold: minimum subspace similarity to consider merge.

        Returns:
            List of (chart_id_a, chart_id_b, similarity_score) tuples.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_merge(
        self,
        chart_a: ChartState,
        chart_b: ChartState,
    ) -> bool:
        """
        Decide whether two charts should be merged.

        Args:
            chart_a: first chart.
            chart_b: second chart.

        Returns:
            True if charts should be merged.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def merge_charts(
        self,
        chart_a: ChartState,
        chart_b: ChartState,
    ) -> ChartState:
        """
        Merge two charts into a single new chart.

        Combines geometry (weighted average of PPCA parameters),
        merges slot lists, and assigns a new chart_id.

        Args:
            chart_a: first chart to merge.
            chart_b: second chart to merge.

        Returns:
            New merged ChartState. Original charts should be retired.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_split(
        self,
        chart_state: ChartState,
        h_assigned: Tensor,
    ) -> bool:
        """
        Decide whether a chart should be split.

        Criteria: bimodal residual distribution, high internal variance,
        excessive support relative to rank.

        Args:
            chart_state: chart to evaluate.
            h_assigned: features assigned to this chart [N, D].

        Returns:
            True if chart should be split.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def split_chart(
        self,
        chart_state: ChartState,
        h_assigned: Tensor,
    ) -> List[ChartState]:
        """
        Split a chart into two sub-charts.

        Uses 2-means in the PPCA latent space to separate modes,
        then fits separate PPCA models to each cluster.

        Args:
            chart_state: chart to split.
            h_assigned: features assigned to this chart [N, D].

        Returns:
            List of two new ChartState objects.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

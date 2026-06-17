"""ChartBuilder: discovers and initializes new charts from uncovered features."""

from typing import List

from torch import Tensor

from .chart_state import ChartState


class ChartBuilder:
    """
    Builds new chart candidates from features not covered by existing charts.

    The builder:
    1. Identifies features not well-explained by existing charts.
    2. Clusters uncovered features into candidate components.
    3. Fits a PPCA model to each component.
    4. Accepts or rejects candidates via MDL criterion.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: chart configuration dict with keys:
                rank, min_support, max_charts_per_layer,
                posterior_threshold, entropy_threshold, mdl_lambda.
        """
        self.config = config
        self.min_support: int = config.get("min_support", 16)
        self.max_charts_per_layer: int = config.get("max_charts_per_layer", 24)
        self.ppca_rank: int = config.get("rank", 8)
        self.mdl_lambda: float = config.get("mdl_lambda", 1.0)
        self.posterior_threshold: float = config.get("posterior_threshold", 0.55)
        self.entropy_threshold: float = config.get("entropy_threshold", 1.0)

    def build_candidates(
        self,
        h_chart: Tensor,
        existing_charts: List[ChartState],
    ) -> List[ChartState]:
        """
        Build candidate charts from features.

        Args:
            h_chart: pre-adapter features of shape [B, D].
            existing_charts: current chart states in this layer.

        Returns:
            List of new candidate ChartState objects.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def split_covered_boundary_uncovered(
        self,
        h_chart: Tensor,
        existing_charts: List[ChartState],
    ):
        """
        Split h_chart into covered, boundary, and uncovered subsets.

        Args:
            h_chart: pre-adapter features of shape [B, D].
            existing_charts: current chart states.

        Returns:
            Tuple of (covered_mask, boundary_mask, uncovered_mask).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def build_candidate_components(self, h_uncovered: Tensor):
        """
        Cluster uncovered features into candidate chart components.

        Args:
            h_uncovered: uncovered features of shape [N, D].

        Returns:
            List of component feature tensors, each of shape [n_i, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def fit_chart_from_component(
        self,
        component_features: Tensor,
        layer_id: int,
        chart_id: int,
    ) -> ChartState:
        """
        Fit a PPCA model to a component and create a ChartState.

        Args:
            component_features: features assigned to this component [n, D].
            layer_id: ViT block index.
            chart_id: unique chart id to assign.

        Returns:
            Initialized ChartState (state='candidate').
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def accept_candidate(
        self,
        candidate: ChartState,
        existing_charts: List[ChartState],
    ) -> bool:
        """
        Decide whether to accept a candidate chart into the atlas.

        Args:
            candidate: proposed new ChartState.
            existing_charts: current charts in the layer.

        Returns:
            True if candidate should be accepted.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_candidate_mdl_gain(
        self,
        candidate: ChartState,
        h_component: Tensor,
        existing_charts: List[ChartState],
    ) -> float:
        """
        Compute MDL gain from adding this candidate.

        Args:
            candidate: proposed chart.
            h_component: features assigned to candidate [n, D].
            existing_charts: current charts.

        Returns:
            MDL gain (positive = better to add chart).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def assign_or_create_chart(
        self,
        h_chart: Tensor,
        existing_charts: List[ChartState],
    ) -> List[ChartState]:
        """
        Assign features to existing charts or create new ones.

        This is the main entry point called per-task after collecting
        chart features for the current task.

        Args:
            h_chart: pre-adapter features of shape [B, D].
            existing_charts: current chart states (may be empty).

        Returns:
            Updated list of ChartState (existing + new).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

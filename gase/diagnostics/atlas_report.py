"""AtlasReporter: summarizes atlas state across layers."""

from typing import Dict, List

from ..atlas.chart_state import ChartState


class AtlasReporter:
    """
    Generates human-readable reports on atlas state.

    Reports include per-layer chart counts, lifecycle states,
    quality metrics, and support statistics.
    """

    def __init__(self, writer=None):
        """
        Args:
            writer: optional TensorBoard/MLflow writer.
        """
        self.writer = writer

    def summarize_layer_atlas(
        self,
        layer_id: int,
        chart_states: List[ChartState],
    ) -> Dict:
        """
        Summarize atlas state for a single layer.

        Args:
            layer_id: ViT block index.
            chart_states: charts in this layer.

        Returns:
            Dict with keys: num_charts, num_active, num_mature,
            total_support, avg_quality, etc.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def summarize_all_layers(
        self,
        layer_charts: Dict[int, List[ChartState]],
    ) -> Dict:
        """
        Summarize atlas state across all layers.

        Args:
            layer_charts: dict mapping layer_id -> list of ChartState.

        Returns:
            Dict with per-layer summaries and global metrics.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def log_chart_table(
        self,
        chart_states: List[ChartState],
        layer_id: int,
    ) -> None:
        """
        Log a formatted table of chart states for one layer.

        Args:
            chart_states: charts in this layer.
            layer_id: ViT block index.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def log_chart_lifecycle(
        self,
        chart_states: List[ChartState],
        layer_id: int,
    ) -> None:
        """
        Log lifecycle state distribution for charts.

        Args:
            chart_states: charts in this layer.
            layer_id: ViT block index.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def export_atlas_state(
        self,
        layer_charts: Dict[int, List[ChartState]],
    ) -> Dict:
        """
        Export full atlas state as serializable dict.

        Args:
            layer_charts: dict mapping layer_id -> list of ChartState.

        Returns:
            Nested dict suitable for JSON serialization.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

"""ChartLifecycleManager: state machine for chart lifecycle transitions."""

from .chart_state import ChartState


class ChartLifecycleManager:
    """
    Manages chart lifecycle state transitions.

    Charts progress through: candidate -> provisional -> active -> mature.
    They may also become saturated or dormant.

    Lifecycle decisions are based on support, age, reuse count,
    and quality metrics accumulated over tasks.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: lifecycle configuration dict with keys:
                active_min_support, mature_min_age, mature_min_reuse,
                dormant_patience, etc.
        """
        self.config = config
        self.active_min_support: int = config.get("active_min_support", 32)
        self.mature_min_age: int = config.get("mature_min_age", 2)
        self.mature_min_reuse: int = config.get("mature_min_reuse", 2)
        self.dormant_patience: int = config.get("dormant_patience", 5)

    def should_create_chart(
        self,
        n_uncovered: int,
        existing_charts: int,
        max_charts: int,
    ) -> bool:
        """
        Decide whether to create a new chart.

        Args:
            n_uncovered: number of uncovered samples.
            existing_charts: current number of charts in this layer.
            max_charts: maximum allowed charts per layer.

        Returns:
            True if a new chart should be created.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_promote_to_active(self, chart_state: ChartState) -> bool:
        """
        Check if chart meets criteria to become active.

        Args:
            chart_state: current chart state.

        Returns:
            True if chart should be promoted to 'active'.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_promote_to_mature(self, chart_state: ChartState) -> bool:
        """
        Check if chart meets criteria to become mature.

        Args:
            chart_state: current chart state.

        Returns:
            True if chart should be promoted to 'mature'.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_mark_saturated(self, chart_state: ChartState) -> bool:
        """
        Check if chart should be marked saturated.

        A saturated chart no longer accepts geometric updates
        but can still serve routing.

        Args:
            chart_state: current chart state.

        Returns:
            True if chart should be marked 'saturated'.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_mark_dormant(self, chart_state: ChartState) -> bool:
        """
        Check if chart should be marked dormant.

        A dormant chart has not been used for many tasks and
        may be candidates for pruning.

        Args:
            chart_state: current chart state.

        Returns:
            True if chart should be marked 'dormant'.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def update_state(self, chart_state: ChartState) -> ChartState:
        """
        Evaluate all transition conditions and update chart state.

        Args:
            chart_state: current chart state.

        Returns:
            Updated ChartState with potentially new state.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

"""SlotBuilder: constructs and updates slots from chart residuals."""

from typing import List, Tuple

from torch import Tensor

from ..atlas.chart_state import ChartState
from .slot_state import SlotState


class SlotBuilder:
    """
    Builds and manages slots within a chart.

    Slots decompose the teacher residual within a chart into
    reusable transformation modes. Each slot captures a specific
    pattern of residual that recurs across tasks.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: slot configuration dict with keys:
                input_rank, output_rank, min_support,
                reuse_threshold, new_slot_threshold.
        """
        self.config = config
        self.input_rank: int = config.get("input_rank", 8)
        self.output_rank: int = config.get("output_rank", 4)
        self.min_support: int = config.get("min_support", 16)
        self.reuse_threshold: float = config.get("reuse_threshold", 0.25)
        self.new_slot_threshold: float = config.get("new_slot_threshold", 0.45)

    def build_or_update_slots_for_chart(
        self,
        chart_state: ChartState,
        h_chart: Tensor,
        delta_teacher: Tensor,
        existing_slots: List[SlotState],
    ) -> List[SlotState]:
        """
        Main entry point: build new slots or update existing ones.

        Args:
            chart_state: the chart to build slots for.
            h_chart: pre-adapter features [N, D] assigned to this chart.
            delta_teacher: teacher residuals [N, D].
            existing_slots: current slots in this chart.

        Returns:
            Updated list of SlotState.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_slot_fit_error(
        self,
        slot_state: SlotState,
        h_chart: Tensor,
        delta_teacher: Tensor,
    ) -> Tensor:
        """
        Compute fit error of an existing slot on new data.

        Error = ||delta_teacher - A_{c,s}(h)||^2

        Args:
            slot_state: existing slot.
            h_chart: features [N, D].
            delta_teacher: teacher residuals [N, D].

        Returns:
            Per-sample fit error of shape [N].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_reuse_slot(self, fit_error: Tensor) -> bool:
        """
        Decide whether an existing slot adequately fits the residual.

        Args:
            fit_error: per-sample fit errors [N].

        Returns:
            True if slot should be reused as-is.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_create_slot(
        self,
        fit_error: Tensor,
        support: int,
    ) -> bool:
        """
        Decide whether to create a new slot for unexplained residual.

        Args:
            fit_error: best-fit error from existing slots [N].
            support: number of samples that would support a new slot.

        Returns:
            True if a new slot should be created.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def create_slot_from_residuals(
        self,
        chart_state: ChartState,
        h_chart: Tensor,
        delta_teacher: Tensor,
        task_id: int,
    ) -> SlotState:
        """
        Create a new slot from unexplained residuals.

        Fits projection bases (P, R) and linear map (B) via
        low-rank regression of delta_teacher on h_chart.

        Args:
            chart_state: parent chart.
            h_chart: features [N, D] needing a new slot.
            delta_teacher: residuals [N, D].
            task_id: current task id.

        Returns:
            New SlotState (state='candidate').
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def update_existing_slot(
        self,
        slot_state: SlotState,
        h_chart: Tensor,
        delta_teacher: Tensor,
    ) -> SlotState:
        """
        Update an existing slot with new data via EMA.

        Args:
            slot_state: existing slot to update.
            h_chart: features [N, D] assigned to this slot.
            delta_teacher: residuals [N, D].

        Returns:
            Updated SlotState.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def estimate_slot_bases(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Estimate P and R bases via SVD of cross-covariance.

        P captures input directions most predictive of delta.
        R captures output directions where delta is largest.

        Args:
            h_chart: features [N, D].
            delta_teacher: residuals [N, D].

        Returns:
            Tuple of (P [D, input_rank], R [output_rank, D]).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_slot_key(
        self,
        h_chart: Tensor,
        P: Tensor,
    ) -> Tensor:
        """
        Compute a prototype key vector for key-based slot routing.

        The key is the mean of P^T @ h_chart, collapsed to a D-dim vector
        via P.

        Args:
            h_chart: features [N, D] assigned to this slot.
            P: input projection basis [D, input_rank].

        Returns:
            Key vector of shape [D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

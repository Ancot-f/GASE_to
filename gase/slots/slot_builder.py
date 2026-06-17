"""SlotBuilder: constructs and updates slots from chart residuals."""

import logging
from typing import List, Tuple

import torch
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

    # ------------------------------------------------------------------
    #  Phase-4: single slot creation
    # ------------------------------------------------------------------

    def create_single_slot_from_residuals(
        self,
        chart_state: ChartState,
        h_chart: Tensor,
        delta_teacher: Tensor,
        task_id: int,
        slot_id: int = 0,
    ) -> SlotState:
        """
        Phase-4: create exactly one slot from all residuals assigned to one chart.

        Fits input basis P and output basis R via cross-covariance SVD,
        computes slot key, and creates a SlotState.

        Args:
            chart_state: parent ChartState (must have mu set).
            h_chart: features of shape [N, D].
            delta_teacher: teacher residuals of shape [N, D].
            task_id: current task id.
            slot_id: slot id (default 0).

        Returns:
            SlotState with P, R, b, key populated.
        """
        P, R = self.estimate_slot_bases(h_chart, delta_teacher)
        b = delta_teacher.mean(dim=0)  # [D]

        # Center h_chart by chart mean for key computation
        h_centered = h_chart - chart_state.mu.unsqueeze(0)
        key = self.compute_slot_key(h_centered, P)

        slot_state = SlotState(
            slot_id=slot_id,
            chart_id=chart_state.chart_id,
            layer_id=chart_state.layer_id,
            input_rank=self.input_rank,
            output_rank=self.output_rank,
            P=P.clone().detach(),
            R=R.clone().detach(),
            B=None,
            b=b.clone().detach(),
            key=key.clone().detach(),
            support=h_chart.shape[0],
            quality={},
            state="active",
            created_task_id=task_id,
            last_updated_task_id=task_id,
            used_count=0,
        )
        chart_state.add_slot_id(slot_id)

        logging.info(
            "[L9Slot] slot_id=%d input_rank=%d output_rank=%d support=%d",
            slot_id, self.input_rank, self.output_rank, h_chart.shape[0],
        )
        return slot_state

    # ------------------------------------------------------------------
    #  Basis estimation
    # ------------------------------------------------------------------

    def estimate_slot_bases(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Estimate residual-sensitive input basis P and output basis R.

        Uses cross-covariance SVD between centered h_chart and delta_teacher:
            M = X^T @ Y / (N-1)
            U, S, Vh = SVD(M)
            P = U[:, :input_rank]
            R = Vh.T[:, :output_rank]

        Args:
            h_chart: features of shape [N, D].
            delta_teacher: teacher residuals of shape [N, D].

        Returns:
            Tuple of (P [D, input_rank], R [D, output_rank]).
        """
        N = h_chart.shape[0]
        X = h_chart - h_chart.mean(dim=0, keepdim=True)       # [N, D]
        Y = delta_teacher - delta_teacher.mean(dim=0, keepdim=True)  # [N, D]

        M = X.mT @ Y / max(N - 1, 1)  # [D, D]

        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        P = U[:, :self.input_rank]              # [D, input_rank]
        R = Vh[:self.output_rank, :]            # [output_rank, D]

        return P, R

    def compute_slot_key(self, h_chart: Tensor, P: Tensor) -> Tensor:
        """
        Compute slot key as mean P-space coordinate.

        Args:
            h_chart: features of shape [N, D] (should be centered by chart mu).
            P: input projection basis of shape [D, input_rank].

        Returns:
            Key vector of shape [input_rank].
        """
        z = h_chart @ P  # [N, input_rank]
        return z.mean(dim=0)  # [input_rank]

    # ------------------------------------------------------------------
    #  Unimplemented (Phase-5+)
    # ------------------------------------------------------------------

    def build_or_update_slots_for_chart(
        self,
        chart_state: ChartState,
        h_chart: Tensor,
        delta_teacher: Tensor,
        existing_slots: List[SlotState],
    ) -> List[SlotState]:
        raise NotImplementedError("Phase-5+ will implement multi-slot logic.")

    def compute_slot_fit_error(
        self, slot_state: SlotState, h_chart: Tensor, delta_teacher: Tensor
    ) -> Tensor:
        raise NotImplementedError("Phase-5+ will implement fit error.")

    def should_reuse_slot(self, fit_error: Tensor) -> bool:
        raise NotImplementedError("Phase-5+ will implement reuse logic.")

    def should_create_slot(self, fit_error: Tensor, support: int) -> bool:
        raise NotImplementedError("Phase-5+ will implement creation logic.")

    def create_slot_from_residuals(
        self,
        chart_state: ChartState,
        h_chart: Tensor,
        delta_teacher: Tensor,
        task_id: int,
    ) -> SlotState:
        raise NotImplementedError("Phase-4 uses create_single_slot_from_residuals.")

    def update_existing_slot(
        self, slot_state: SlotState, h_chart: Tensor, delta_teacher: Tensor
    ) -> SlotState:
        raise NotImplementedError("Phase-5+ will implement EMA update.")

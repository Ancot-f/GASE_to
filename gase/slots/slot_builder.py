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
        # Anti-corruption: slot must not overwrite existing slot_id
        assert slot_id not in chart_state.slot_ids, (
            f"Slot {slot_id} already exists in chart {chart_state.chart_id} "
            f"layer {chart_state.layer_id}. Old slots must not be overwritten."
        )
        assert delta_teacher is not None, "SlotBuilder requires delta_teacher."
        assert h_chart.shape == delta_teacher.shape, (
            f"Shape mismatch: h_chart {h_chart.shape} vs delta_teacher {delta_teacher.shape}"
        )
        assert chart_state.mu is not None, "ChartState must have mu set before slot creation."

        P, R = self.estimate_slot_bases(h_chart, delta_teacher)
        b = delta_teacher.mean(dim=0)  # [D]

        # key and key_var from P-space projection
        h_centered = h_chart - chart_state.mu.unsqueeze(0)
        key, key_var = self.compute_slot_key_with_var(h_centered, P)

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
            key_var=key_var.clone().detach(),
            support=h_chart.shape[0],
            quality={},
            state="active",
            created_task_id=task_id,
            last_updated_task_id=task_id,
            used_count=0,
        )
        chart_state.add_slot_id(slot_id)

        logging.info(
            "[SlotContract] layer=%d chart=%d slot=%d "
            "definition=residual_field_mode method=cross_covariance_svd "
            "P_shape=%s R_shape=%s key_norm=%.4f b_norm=%.4f support=%d",
            chart_state.layer_id, chart_state.chart_id, slot_id,
            list(P.shape), list(R.shape),
            float(key.norm()), float(b.norm()), h_chart.shape[0],
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
        """Legacy: returns key only."""
        z = h_chart @ P
        return z.mean(dim=0)

    def compute_slot_key_with_var(
        self, h_chart: Tensor, P: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute slot key and key variance in P-space.

        key = mean(P^T @ (h - mu)), key_var = var(P^T @ (h - mu)) + eps.
        """
        z = h_chart @ P
        key = z.mean(dim=0)
        key_var = z.var(dim=0, unbiased=False) + 1e-6
        return key, key_var

    # ------------------------------------------------------------------
    #  Future: slot compatibility evaluation (skeleton only)
    # ------------------------------------------------------------------

    def evaluate_slot_compatibility(
        self,
        chart_state: ChartState,
        existing_slots: List[SlotState],
        h_chart: Tensor,
        delta_teacher: Tensor,
    ):
        """
        Future only. Check whether current residual field can reuse an existing slot.
        Do not use in Phase-6.5.
        """
        raise NotImplementedError("Phase-7+ will implement slot reuse evaluation.")

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

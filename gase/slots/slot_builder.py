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

        # key and key_var from P-space projection (legacy, for ablation)
        h_centered = h_chart - chart_state.mu.unsqueeze(0)
        key, key_var = self.compute_slot_key_with_var(h_centered, P)

        # Phase-7.5: router_key in shared Q-space
        router_key, router_var = self.compute_router_key(h_chart, chart_state)

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
            router_key=router_key.clone().detach(),
            router_var=router_var.clone().detach(),
            router_support=h_chart.shape[0],
            support=h_chart.shape[0],
            quality={},
            state="active",
            created_task_id=task_id,
            last_updated_task_id=task_id,
            used_count=0,
        )
        chart_state.add_slot_id(slot_id)

        # Phase-9: compute self-NLL stats for router calibration
        nll_stats = _compute_self_nll_stats(h_chart, chart_state, slot_state)
        slot_state.router_nll_mean = nll_stats["mean"]
        slot_state.router_nll_std = nll_stats["std"]
        slot_state.router_nll_q90 = nll_stats["q90"]
        slot_state.router_nll_q95 = nll_stats["q95"]
        slot_state.router_logdet = nll_stats["logdet"]
        slot_state.router_support = h_chart.shape[0]

        logging.info(
            "[SlotContract] layer=%d chart=%d slot=%d "
            "definition=residual_field_mode method=cross_covariance_svd "
            "P_shape=%s R_shape=%s "
            "adapter_basis=P_cross_cov router_basis=shared_Q "
            "adapter_key_norm=%.4f router_key_norm=%.4f router_var_mean=%.4f "
            "b_norm=%.4f support=%d",
            chart_state.layer_id, chart_state.chart_id, slot_id,
            list(P.shape), list(R.shape),
            float(key.norm()), float(router_key.norm()), float(router_var.mean()),
            float(b.norm()), h_chart.shape[0],
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

    def compute_router_key(
        self, h_chart: Tensor, chart_state: ChartState,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute routing key in shared chart routing basis Q_router.

        Q_router defaults to chart.U (shared tangent coordinate).
        All slots share the same Q_router, making distances comparable.
        """
        Q = chart_state.Q_router
        if Q is None:
            Q = chart_state.U
            chart_state.Q_router = Q
            chart_state.router_rank = Q.shape[1] if Q is not None else 0
        if Q is None:
            return torch.zeros(1, device=h_chart.device), torch.ones(1, device=h_chart.device)
        X = h_chart - chart_state.mu.unsqueeze(0)
        z = X @ Q
        key = z.mean(dim=0)
        var = z.var(dim=0, unbiased=False) + 1e-6
        return key, var

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


def _compute_self_nll_stats(h_chart, chart_state, slot_state, eps=1e-6):
    """Compute self-NLL stats for router calibration (Phase-9)."""
    import torch
    Q = chart_state.Q_router if getattr(chart_state, "Q_router", None) is not None else getattr(chart_state, "U", None)
    if Q is None:
        return {"mean": 0.0, "std": 1.0, "q90": 0.0, "q95": 0.0, "logdet": 0.0}
    X = h_chart - chart_state.mu.unsqueeze(0)
    z = X @ Q.to(h_chart.device)
    key = slot_state.router_key.to(h_chart.device)
    var = slot_state.router_var.clamp_min(eps).to(h_chart.device)
    maha = ((z - key.unsqueeze(0)) ** 2 / var.unsqueeze(0)).sum(dim=-1)
    logdet = torch.log(var).sum()
    nll = 0.5 * maha + 0.5 * logdet
    result = {
        "mean": float(nll.mean()), "std": float(nll.std(unbiased=False)),
        "q90": float(torch.quantile(nll, 0.90)), "q95": float(torch.quantile(nll, 0.95)),
        "logdet": float(logdet),
    }
    logging.info("[RouterSelfNLL] layer=%d slot=%d support=%d mean=%.2f std=%.2f q50=%.2f q90=%.2f q95=%.2f logdet=%.2f",
                 chart_state.layer_id, slot_state.slot_id, h_chart.shape[0],
                 result["mean"], result["std"],
                 float(torch.quantile(nll, 0.50)), result["q90"], result["q95"], result["logdet"])
    return result

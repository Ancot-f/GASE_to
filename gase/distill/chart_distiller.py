"""ChartAdapterDistiller: distills task-adapter residuals into chart-adapters."""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..atlas.chart_state import ChartState
from ..slots.slot_state import SlotState
from ..adapters.chart_adapter import LinearChartAdapter


class ChartAdapterDistiller:
    """
    Distills task-adapter teacher residuals into chart-slot adapters.

    Phase-4: fits one LinearChartAdapter for one (chart, slot) pair
    via ridge regression in the low-rank latent space.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: distillation config dict.
        """
        self.config = config

    # ------------------------------------------------------------------
    #  Phase-4: single chart-slot ridge fitting
    # ------------------------------------------------------------------

    def fit_linear_chart_adapter(
        self,
        chart_state: ChartState,
        slot_state: SlotState,
        h_chart: Tensor,
        delta_teacher: Tensor,
    ) -> Tuple["LinearChartAdapter", Dict[str, float]]:
        """
        Fit LinearChartAdapter for one chart-slot pair via ridge regression.

        Formula: A(h) = b + R @ B @ P^T @ (h - mu)

        Steps:
          1. Center h by chart mu, project to P-space: Z = (h-mu) @ P  [N, rp]
          2. Center delta by slot b: Y = delta - b                     [N, D]
          3. Project Y to R-space: Y_low = Y @ R                        [N, ro]
          4. Ridge: Z @ B^T ≈ Y_low  =>  B = solve(Z^T Z + λI, Z^T Y_low)^T
          5. Build LinearChartAdapter with (P, R, B, b)

        Args:
            chart_state: ChartState with mu.
            slot_state: SlotState with P, R, b.
            h_chart: features of shape [N, D].
            delta_teacher: teacher residuals of shape [N, D].

        Returns:
            Tuple of (LinearChartAdapter, metrics dict).
        """
        N, D = h_chart.shape
        mu = chart_state.mu.to(h_chart.device)
        P = slot_state.P.to(h_chart.device)  # [D, rp]
        R = slot_state.R.to(h_chart.device)  # [D, ro]
        b = slot_state.b.to(h_chart.device)  # [D]

        rp = P.shape[1]  # input_rank
        ro = R.shape[1]  # output_rank

        # Center
        X = h_chart - mu.unsqueeze(0)         # [N, D]
        Z = X @ P                              # [N, rp]   (P: [D, rp])
        Y = delta_teacher - b.unsqueeze(0)    # [N, D]
        Y_low = Y @ R.mT                       # [N, ro]   (R: [ro, D], R.mT: [D, ro])

        # Ridge regression: Z @ B^T ≈ Y_low
        lambda_ridge = self.config.get("ridge_lambda", 1e-3)
        A = Z.mT @ Z + lambda_ridge * torch.eye(rp, device=h_chart.device, dtype=h_chart.dtype)
        C = Z.mT @ Y_low
        B_T = torch.linalg.solve(A, C)        # [rp, ro]
        B = B_T.T                              # [ro, rp]

        # Build adapter
        adapter = LinearChartAdapter(
            dim=D,
            input_rank=rp,
            output_rank=ro,
            chart_id=chart_state.chart_id,
            slot_id=slot_state.slot_id,
            layer_id=chart_state.layer_id,
        )
        adapter.to(h_chart.device)
        adapter.set_projection_bases(P, R)
        adapter.set_linear_map(B, b)

        # Compute predicted residual and metrics
        delta_pred = adapter(h_chart, mu)     # [N, D]

        metrics = _compute_residual_metrics(delta_pred, delta_teacher)

        logging.info(
            "[L%dDistill] residual_mse=%.6f residual_cos=%.4f fit_r2=%.4f norm_ratio=%.4f",
            chart_state.layer_id,
            metrics["residual_mse"],
            metrics["residual_cos"],
            metrics["fit_r2"],
            metrics["norm_ratio"],
        )

        return adapter, metrics

    # ------------------------------------------------------------------
    #  Unimplemented (Phase-5+)
    # ------------------------------------------------------------------

    def distill_for_layer(
        self,
        layer_id: int,
        layer_cache,
        chart_states: List[ChartState],
        slot_states: Dict[int, List[SlotState]],
    ) -> None:
        raise NotImplementedError("Phase-5+ will implement multi-chart-slot distillation.")

    def distill_for_chart_slot(
        self,
        chart_state: ChartState,
        slot_state: SlotState,
        h_chart: Tensor,
        delta_teacher: Tensor,
    ) -> None:
        raise NotImplementedError("Phase-4 uses fit_linear_chart_adapter.")

    def fit_mlp_chart_adapter(
        self, h_chart: Tensor, delta_teacher: Tensor, P: Tensor, R: Tensor
    ) -> None:
        raise NotImplementedError("Phase-5+ will implement MLP fitting.")

    def compute_projection_bases(
        self, h_chart: Tensor, delta_teacher: Tensor, input_rank: int, output_rank: int
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Phase-4 uses SlotBuilder.estimate_slot_bases.")

    def compute_distill_losses(
        self, delta_chart: Tensor, delta_teacher: Tensor
    ) -> Dict[str, Tensor]:
        raise NotImplementedError("Phase-5+ will implement multi-loss distillation.")


def _compute_residual_metrics(delta_pred: Tensor, delta_teacher: Tensor) -> Dict[str, float]:
    """Compute residual fit metrics between predicted and teacher residuals."""
    # MSE
    residual_mse = float(F.mse_loss(delta_pred, delta_teacher))

    # Cosine similarity
    residual_cos = float(F.cosine_similarity(delta_pred, delta_teacher, dim=-1).mean())

    # R^2
    ss_res = ((delta_teacher - delta_pred) ** 2).sum()
    ss_tot = ((delta_teacher - delta_teacher.mean(dim=0, keepdim=True)) ** 2).sum()
    fit_r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    # Norm ratio
    pred_norm = delta_pred.norm(dim=-1).mean()
    teacher_norm = delta_teacher.norm(dim=-1).mean()
    norm_ratio = float(pred_norm / (teacher_norm + 1e-8))

    return {
        "residual_mse": residual_mse,
        "residual_cos": residual_cos,
        "fit_r2": fit_r2,
        "norm_ratio": norm_ratio,
        "delta_teacher_norm": float(teacher_norm),
        "delta_pred_norm": float(pred_norm),
    }

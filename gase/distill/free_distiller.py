"""FreeAdapterDistiller: minimal free-adapter fitting for Phase-6.5."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, optim

from ..adapters.free_adapter import FreeAdapter


class FreeAdapterDistiller:
    """Trains a free-adapter to absorb residual leftover from chart-adapter."""

    def __init__(self, config: dict):
        self.config = config
        self.enabled: bool = config.get("free_adapter", {}).get("enabled", True)

    def fit_free_adapter_for_layer_slot(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        delta_chart: Tensor,
    ) -> Tuple[FreeAdapter, Dict[str, float]]:
        """
        Train a FreeAdapter to fit: target = delta_teacher - delta_chart.

        Args:
            h_chart: features [N, D].
            delta_teacher: teacher residuals [N, D].
            delta_chart: chart-adapter residuals [N, D].

        Returns:
            Tuple of (FreeAdapter, metrics dict).
        """
        free_cfg = self.config.get("free_adapter", {})
        D = h_chart.shape[1]
        adapter = FreeAdapter(D, bottleneck_dim=free_cfg.get("bottleneck_dim", 16),
                              dropout=free_cfg.get("dropout", 0.0), scale=free_cfg.get("scale", 1.0))
        adapter.to(h_chart.device)

        target = delta_teacher - delta_chart.detach()
        epochs = free_cfg.get("epochs", 3)
        lr = free_cfg.get("lr", 0.001)

        opt = optim.AdamW(adapter.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            pred = adapter(h_chart)
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()

        # Compute metrics
        with torch.no_grad():
            free_pred = adapter(h_chart)
            free_mse = float(F.mse_loss(free_pred, target))
            leftover_norm = float(target.norm(dim=-1).mean())
            free_pred_norm = float(free_pred.norm(dim=-1).mean())

            combined = delta_chart + free_pred
            combined_mse = float(F.mse_loss(combined, delta_teacher))
            combined_cos = float(F.cosine_similarity(combined, delta_teacher, dim=-1).mean())
            ss_res = ((delta_teacher - combined) ** 2).sum()
            ss_tot = ((delta_teacher - delta_teacher.mean(dim=0, keepdim=True)) ** 2).sum()
            combined_r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        metrics = {
            "free_mse": free_mse, "leftover_norm": leftover_norm,
            "free_pred_norm": free_pred_norm, "combined_mse": combined_mse,
            "combined_cos": combined_cos, "combined_r2": combined_r2,
        }
        logging.info("[L%dFree] combined_cos=%.4f combined_r2=%.4f free_mse=%.6f",
                     0, combined_cos, combined_r2, free_mse)
        return adapter, metrics

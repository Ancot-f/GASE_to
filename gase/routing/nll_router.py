"""
CalibratedNLLSlotRouter: Gaussian NLL-based slot routing (Phase-9).

Supports: raw NLL, calibrated NLL (z-normalized), prior correction,
and slot0 penalty ablation.
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class CalibratedNLLSlotRouter:
    """
    Slot router using Gaussian NLL in shared Q-space.

    nll_s(h) = 0.5 * maha + 0.5 * logdet
    calibrated_nll_s = (nll_s - self_mean_s) / (self_std_s + eps)
    score_s = -calibrated_nll_s / T + prior * log(prior_s)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-6,
        use_logdet: bool = True,
        calibrate_nll: bool = False,
        prior_mode: str = "uniform",
        prior_weight: float = 0.0,
        slot0_penalty: float = 0.0,
    ):
        self.temperature = temperature
        self.eps = eps
        self.use_logdet = use_logdet
        self.calibrate_nll = calibrate_nll
        self.prior_mode = prior_mode
        self.prior_weight = prior_weight
        self.slot0_penalty = slot0_penalty

    def compute_nll(self, h_chart: Tensor, chart_state, slot) -> Tensor:
        """Gaussian NLL in shared Q-space. Returns [B]."""
        Q = getattr(chart_state, "Q_router", None)
        if Q is None:
            Q = getattr(chart_state, "U", None)
        if Q is None:
            return torch.zeros(h_chart.shape[0], device=h_chart.device)
        Q = Q.to(h_chart.device)
        X = h_chart - chart_state.mu.to(h_chart.device).unsqueeze(0)
        z = X @ Q
        key = slot.router_key.to(h_chart.device)
        var = slot.router_var.clamp_min(self.eps).to(h_chart.device)
        maha = ((z - key.unsqueeze(0)) ** 2 / var.unsqueeze(0)).sum(dim=-1)
        if self.use_logdet:
            logdet = torch.log(var).sum()
        else:
            logdet = 0.0
        return 0.5 * maha + 0.5 * logdet

    def compute_slot_distances(
        self, h_chart: Tensor, chart_state, slot_states: Dict[int, object],
    ) -> Tuple[Tensor, List[int]]:
        """Compute NLL-based scores (higher = better). Returns scores [B,S], slot_ids."""
        slot_ids = sorted(slot_states.keys())
        if not slot_ids:
            return torch.zeros(h_chart.shape[0], 0, device=h_chart.device), []
        scores_list = []
        num_slots = len(slot_ids)

        for sid in slot_ids:
            ss = slot_states[sid]
            nll = self.compute_nll(h_chart, chart_state, ss)  # [B]
            if self.calibrate_nll and ss.router_nll_mean is not None and ss.router_nll_std is not None and ss.router_nll_std > max(self.eps, 0.1):
                calib = (nll - ss.router_nll_mean) / ss.router_nll_std
            else:
                calib = nll  # fallback to raw NLL when std too small or stats unavailable
            score = -calib / self.temperature
            if self.slot0_penalty > 0 and sid == 0:
                score = score - self.slot0_penalty
            if self.prior_weight > 0:
                if self.prior_mode == "uniform":
                    prior = 1.0 / max(num_slots, 1)
                elif self.prior_mode == "support":
                    sup = getattr(ss, "router_support", 1) or 1
                    total_sup = sum(getattr(slot_states[s], "router_support", 1) or 1 for s in slot_ids)
                    prior = sup / max(total_sup, 1)
                else:
                    prior = 1.0 / max(num_slots, 1)
                score = score + self.prior_weight * float(torch.log(torch.tensor(prior + self.eps)))
            scores_list.append(score)

        scores = torch.stack(scores_list, dim=1)  # [B, S]
        return scores, slot_ids

    def route(self, h_chart: Tensor, chart_state, slot_states: Dict[int, object]) -> Dict[str, Any]:
        """Per-sample routing using NLL scores."""
        B = h_chart.shape[0]
        scores, slot_id_list = self.compute_slot_distances(h_chart, chart_state, slot_states)
        S = len(slot_id_list)

        if S == 0:
            return {"slot_ids": torch.zeros(B, dtype=torch.long, device=h_chart.device),
                    "scores": scores, "slot_id_list": [], "entropy": torch.zeros(B, device=h_chart.device),
                    "margin": torch.full([B], float("inf"), device=h_chart.device)}

        probs = F.softmax(scores, dim=1)
        selected_idx = scores.argmax(dim=1)
        selected_slot_ids = torch.tensor([slot_id_list[i.item()] for i in selected_idx],
                                         device=h_chart.device, dtype=torch.long)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        if S > 1:
            top2 = scores.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.full([B], float("inf"), device=h_chart.device)

        return {"slot_ids": selected_slot_ids, "scores": scores, "slot_id_list": slot_id_list,
                "entropy": entropy, "margin": margin, "probs": probs}

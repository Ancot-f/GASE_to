"""
KeySlotRouter: Per-sample task-agnostic slot routing via Mahalanobis key distance.

Uses only h_chart and slot key statistics (key, key_var).
No labels, task_id, logits, or classifier margin.
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class KeySlotRouter:
    """
    Task-agnostic slot router based on Mahalanobis distance to slot keys.

    For each sample, computes distance to each slot key in P-space,
    selects the nearest slot. Supports both Mahalanobis (key_var) and
    Euclidean distance.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        use_mahalanobis: bool = True,
        eps: float = 1e-6,
        quality_weight: float = 0.0,
    ):
        self.temperature = temperature
        self.use_mahalanobis = use_mahalanobis
        self.eps = eps
        self.quality_weight = quality_weight

    def compute_slot_distances(
        self,
        h_chart: Tensor,
        chart_state: object,
        slot_states: Dict[int, object],
        use_shared_router_basis: bool = True,
    ) -> Tuple[Tensor, List[int]]:
        """
        Compute distance from each sample to each slot.

        Phase-7.5 default: shared Q-space (router_key/router_var).
        Fallback: per-slot P-space (key/key_var).

        Args:
            h_chart: features [B, D].
            chart_state: ChartState with mu and Q_router.
            slot_states: dict slot_id -> SlotState.
            use_shared_router_basis: if True, use Q_router space.

        Returns:
            distances: [B, S], slot_ids: list of slot ids.
        """
        slot_ids = sorted(slot_states.keys())
        if not slot_ids:
            return torch.zeros(h_chart.shape[0], 0, device=h_chart.device), []

        mu = chart_state.mu.to(h_chart.device)
        X = h_chart - mu.unsqueeze(0)  # [B, D]

        if use_shared_router_basis:
            Q = getattr(chart_state, "Q_router", None)
            if Q is None:
                Q = getattr(chart_state, "U", None)
            if Q is None:
                return torch.zeros(h_chart.shape[0], len(slot_ids), device=h_chart.device), slot_ids
            Q = Q.to(h_chart.device)
            z = X @ Q  # [B, r_q]

            distances = []
            for sid in slot_ids:
                ss = slot_states[sid]
                rk = getattr(ss, "router_key", None)
                rv = getattr(ss, "router_var", None)
                if rk is None:
                    rk = ss.key.to(h_chart.device)
                    rv = ss.key_var.to(h_chart.device) if ss.key_var is not None else None
                else:
                    rk = rk.to(h_chart.device)
                    rv = rv.to(h_chart.device) if rv is not None else None

                if self.use_mahalanobis and rv is not None:
                    d = ((z - rk.unsqueeze(0)) ** 2 / (rv.unsqueeze(0) + self.eps)).sum(dim=-1)
                else:
                    d = ((z - rk.unsqueeze(0)) ** 2).sum(dim=-1)
                distances.append(d)
        else:
            distances = []
            for sid in slot_ids:
                ss = slot_states[sid]
                P = ss.P.to(h_chart.device)
                key = ss.key.to(h_chart.device)
                key_var = ss.key_var.to(h_chart.device) if ss.key_var is not None else None
                z = X @ P
                if self.use_mahalanobis and key_var is not None:
                    d = ((z - key.unsqueeze(0)) ** 2 / (key_var.unsqueeze(0) + self.eps)).sum(dim=-1)
                else:
                    d = ((z - key.unsqueeze(0)) ** 2).sum(dim=-1)
                distances.append(d)

        distances = torch.stack(distances, dim=1)  # [B, S]
        return distances, slot_ids

    def route(
        self,
        h_chart: Tensor,
        chart_state: object,
        slot_states: Dict[int, object],
    ) -> Dict[str, Any]:
        """
        Per-sample slot routing.

        Args:
            h_chart: features [B, D].
            chart_state: ChartState.
            slot_states: dict slot_id -> SlotState.

        Returns:
            Dict with slot_ids [B], distances [B,S], scores [B,S],
            entropy [B], margin [B], slot_id_list, probs [B,S].
        """
        B = h_chart.shape[0]
        distances, slot_id_list = self.compute_slot_distances(h_chart, chart_state, slot_states)
        S = len(slot_id_list)

        if S == 0:
            return {
                "slot_ids": torch.zeros(B, dtype=torch.long, device=h_chart.device),
                "distances": distances,
                "scores": torch.zeros(B, 0, device=h_chart.device),
                "entropy": torch.zeros(B, device=h_chart.device),
                "margin": torch.full([B], float("inf"), device=h_chart.device),
                "slot_id_list": [],
                "probs": torch.zeros(B, 0, device=h_chart.device),
            }

        scores = -distances / self.temperature  # [B, S]
        probs = F.softmax(scores, dim=1)         # [B, S]
        selected_idx = scores.argmax(dim=1)       # [B]

        slot_ids_list = [slot_id_list[i.item()] for i in selected_idx]
        selected_slot_ids = torch.tensor(slot_ids_list, device=h_chart.device, dtype=torch.long)

        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # [B]

        if S > 1:
            top2 = scores.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.full([B], float("inf"), device=h_chart.device)

        return {
            "slot_ids": selected_slot_ids,
            "distances": distances,
            "scores": scores,
            "entropy": entropy,
            "margin": margin,
            "slot_id_list": slot_id_list,
            "probs": probs,
        }

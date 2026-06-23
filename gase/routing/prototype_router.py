"""PrototypeNLLSlotRouter: multi-prototype deploy-prefix slot routing."""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class PrototypeNLLSlotRouter:
    """
    Route slots with a mixture of diagonal Gaussian prototypes in chart Q-space.

    Each slot may store K prototype means/variances learned from deploy-visible
    h_chart features. The slot score is log p(h | slot), approximated by a
    prototype mixture. Higher score is better.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-6,
        use_logdet: bool = True,
        use_proto_prior: bool = True,
        aggregate: str = "logsumexp",
    ):
        self.temperature = temperature
        self.eps = eps
        self.use_logdet = use_logdet
        self.use_proto_prior = use_proto_prior
        self.aggregate = aggregate

    def _project(self, h_chart: Tensor, chart_state) -> Tensor:
        Q = getattr(chart_state, "Q_router", None)
        if Q is None:
            Q = getattr(chart_state, "U", None)
        if Q is None:
            return torch.zeros(h_chart.shape[0], 1, device=h_chart.device, dtype=h_chart.dtype)
        X = h_chart - chart_state.mu.to(h_chart.device).unsqueeze(0)
        return X @ Q.to(h_chart.device)

    def compute_slot_scores(
        self, h_chart: Tensor, chart_state, slot_states: Dict[int, object],
    ) -> Tuple[Tensor, List[int]]:
        slot_ids = sorted(slot_states.keys())
        if not slot_ids:
            return torch.zeros(h_chart.shape[0], 0, device=h_chart.device), []

        z = self._project(h_chart, chart_state)
        scores = []
        for sid in slot_ids:
            ss = slot_states[sid]
            proto_key = getattr(ss, "router_proto_key", None)
            proto_var = getattr(ss, "router_proto_var", None)
            proto_count = getattr(ss, "router_proto_count", None)

            if proto_key is None or proto_var is None:
                proto_key = ss.router_key.unsqueeze(0)
                proto_var = ss.router_var.unsqueeze(0)
                proto_count = torch.ones(1, device=h_chart.device, dtype=h_chart.dtype)

            proto_key = proto_key.to(h_chart.device)
            proto_var = proto_var.clamp_min(self.eps).to(h_chart.device)
            proto_count = proto_count.to(h_chart.device) if proto_count is not None else None

            diff = z.unsqueeze(1) - proto_key.unsqueeze(0)  # [B, K, r]
            maha = (diff.pow(2) / proto_var.unsqueeze(0)).sum(dim=-1)
            if self.use_logdet:
                logdet = torch.log(proto_var).sum(dim=-1).unsqueeze(0)
            else:
                logdet = 0.0
            nll = 0.5 * maha + 0.5 * logdet

            proto_logp = -nll
            if self.use_proto_prior and proto_count is not None:
                prior = proto_count.clamp_min(self.eps)
                prior = prior / prior.sum().clamp_min(self.eps)
                proto_logp = proto_logp + torch.log(prior).unsqueeze(0)

            if self.aggregate == "max":
                score = proto_logp.max(dim=1).values
            else:
                score = torch.logsumexp(proto_logp, dim=1)
            scores.append(score / self.temperature)

        return torch.stack(scores, dim=1), slot_ids

    def route(self, h_chart: Tensor, chart_state, slot_states: Dict[int, object]) -> Dict[str, Any]:
        B = h_chart.shape[0]
        scores, slot_id_list = self.compute_slot_scores(h_chart, chart_state, slot_states)
        S = len(slot_id_list)
        if S == 0:
            return {"slot_ids": torch.zeros(B, dtype=torch.long, device=h_chart.device),
                    "scores": scores, "slot_id_list": [], "entropy": torch.zeros(B, device=h_chart.device),
                    "margin": torch.full([B], float("inf"), device=h_chart.device),
                    "probs": torch.zeros(B, 0, device=h_chart.device)}

        probs = F.softmax(scores, dim=1)
        selected_idx = scores.argmax(dim=1)
        selected = torch.tensor([slot_id_list[i.item()] for i in selected_idx],
                                device=h_chart.device, dtype=torch.long)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        if S > 1:
            top2 = scores.topk(k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.full([B], float("inf"), device=h_chart.device)
        return {"slot_ids": selected, "scores": scores, "slot_id_list": slot_id_list,
                "entropy": entropy, "margin": margin, "probs": probs}

    def topm_slot_ids(self, h_chart: Tensor, chart_state, slot_states: Dict[int, object],
                      m: int = 3) -> Tuple[Tensor, List[int], Tensor]:
        scores, slot_id_list = self.compute_slot_scores(h_chart, chart_state, slot_states)
        if not slot_id_list:
            empty = torch.zeros(h_chart.shape[0], 0, dtype=torch.long, device=h_chart.device)
            return empty, [], scores
        k = min(max(1, m), len(slot_id_list))
        top_idx = scores.topk(k=k, dim=1).indices
        id_tensor = torch.tensor(slot_id_list, dtype=torch.long, device=h_chart.device)
        return id_tensor[top_idx], slot_id_list, scores

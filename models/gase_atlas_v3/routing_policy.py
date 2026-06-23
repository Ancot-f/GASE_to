"""Shared GASE-Atlas v3 routing and free-trigger policy.

This module is intentionally small: training and inference both call the same
policy so free-adapter usage is not decided by two drifting implementations.
"""

from dataclasses import dataclass

import torch


@dataclass
class SlotCandidate:
    chart: object
    adapter: object
    chart_id: int
    slot_id: int
    score: float
    d2: float
    geo_score: float
    full_r2: float
    subspace_r2: float
    adapter_cos: float
    norm_ratio: float
    conflict: float
    radius_ratio: float


@dataclass
class RouteDecision:
    use_chart: bool
    use_free: bool
    use_identity: bool
    reason: str
    candidate: SlotCandidate | None = None
    candidates: object = None
    margin: float = float("inf")
    entropy: float = 0.0
    weights: torch.Tensor | None = None


class GASEAtlasRoutingPolicy:
    """Quality-gated chart/slot routing.

    The policy first enumerates valid chart-slot candidates, applies hard masks,
    then returns a chart decision or free/identity fallback.
    """

    def __init__(
        self,
        temperature=1.0,
        min_r2=0.25,
        min_adapter_cos=0.20,
        max_norm_ratio=2.0,
        uncertainty_margin=0.5,
        uncertainty_entropy=1.2,
        beta_r2=3.0,
        beta_conflict=0.5,
        l11_disable_chart=True,
        l11_use_free=False,
    ):
        self.temperature = max(float(temperature), 1e-6)
        self.min_r2 = float(min_r2)
        self.min_adapter_cos = float(min_adapter_cos)
        self.max_norm_ratio = float(max_norm_ratio)
        self.uncertainty_margin = float(uncertainty_margin)
        self.uncertainty_entropy = float(uncertainty_entropy)
        self.beta_r2 = float(beta_r2)
        self.beta_conflict = float(beta_conflict)
        self.l11_disable_chart = bool(l11_disable_chart)
        self.l11_use_free = bool(l11_use_free)

    def decide(self, h, charts, layer_id, top_k=1):
        if int(layer_id) == 11 and self.l11_disable_chart:
            return RouteDecision(
                use_chart=False,
                use_free=self.l11_use_free,
                use_identity=not self.l11_use_free,
                reason="l11_protect",
            )

        candidates = self.candidates(h, charts)
        if not candidates:
            if int(layer_id) == 11 and not self.l11_use_free:
                return RouteDecision(False, False, True, "no_valid_slot_l11_identity")
            return RouteDecision(False, True, False, "no_valid_slot")

        candidates.sort(key=lambda c: c.score, reverse=True)
        selected = candidates[:max(1, min(int(top_k), len(candidates)))]
        best = selected[0]
        scores = torch.tensor([c.score for c in selected], device=h.device)
        weights = torch.softmax(scores, dim=0)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum().item()
        margin = best.score - candidates[1].score if len(candidates) > 1 else float("inf")

        if margin < self.uncertainty_margin or entropy > self.uncertainty_entropy:
            return RouteDecision(
                use_chart=True,
                use_free=False,
                use_identity=False,
                reason="mixed_uncertain",
                candidate=best,
                candidates=selected,
                margin=margin,
                entropy=entropy,
                weights=weights,
            )

        return RouteDecision(
            use_chart=True,
            use_free=False,
            use_identity=False,
            reason="chart",
            candidate=best,
            candidates=selected,
            margin=margin,
            entropy=entropy,
            weights=weights,
        )

    def free_mask(self, features, charts, layer_id):
        mask = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        for i in range(features.shape[0]):
            decision = self.decide(features[i:i + 1], charts, layer_id)
            mask[i] = decision.use_free
        return mask

    def candidates(self, h, charts):
        results = []
        for chart in charts:
            if getattr(chart, "status", "inactive") != "active":
                continue
            if getattr(chart, "num_adapters", 0) == 0:
                continue

            d2 = chart.mahalanobis_d2(h).item()
            raw_radius = max(float(chart.radius_d2.item()), 1e-6)
            radius_ratio = float(d2 / raw_radius)

            chart_r2 = float(max(
                getattr(chart, "full_r2", 0.0),
                getattr(chart, "subspace_r2", 0.0),
            ))
            if chart_r2 < self.min_r2:
                continue

            conflict = float(getattr(chart, "overlap_rate", 0.0))
            geo_score = -d2 / self.temperature

            for slot_id, adapter in chart._adapters.items():
                full_r2 = float(getattr(adapter, "full_r2", getattr(chart, "full_r2", 0.0)))
                sub_r2 = float(getattr(adapter, "subspace_r2", getattr(chart, "subspace_r2", 0.0)))
                adapter_cos = float(getattr(adapter, "adapter_cos", getattr(chart, "adapter_cos", 1.0)))
                norm_ratio = float(getattr(adapter, "norm_ratio", 0.0))

                if sub_r2 < self.min_r2:
                    continue
                if adapter_cos < self.min_adapter_cos:
                    continue
                if self.max_norm_ratio > 0.0 and norm_ratio > self.max_norm_ratio:
                    continue

                score = geo_score + self.beta_r2 * sub_r2 - self.beta_conflict * conflict
                results.append(SlotCandidate(
                    chart=chart, adapter=adapter,
                    chart_id=int(chart.chart_id), slot_id=int(slot_id),
                    score=float(score), d2=float(d2),
                    geo_score=float(geo_score),
                    full_r2=full_r2, subspace_r2=sub_r2,
                    adapter_cos=adapter_cos, norm_ratio=norm_ratio,
                    conflict=conflict, radius_ratio=radius_ratio,
                ))
        return results

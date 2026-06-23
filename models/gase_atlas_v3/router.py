"""ChartRouterV3 -routes features to charts with structured RouteResult.

Implements diagonal Mahalanobis distance routing with per-sample diagnostics:
margin, entropy, uncertainty flag. Supports top-k mixture for L9/L10 and
conservative top-1 for L11.

Corresponds to design doc Section 6.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RouteResult:
    """Structured routing output for a single sample (design doc Section 6.3)."""
    selected_chart: int | None       # selected chart ID
    selected_adapter: int | None     # selected adapter slot
    top1_distance: float             # Mahalanobis d虏 to nearest chart
    top2_distance: float             # Mahalanobis d虏 to second-nearest
    margin: float                    # d_second - d_first
    entropy: float                   # softmax entropy over all charts
    weights: torch.Tensor            # [num_charts] normalized routing weights
    uncertain: bool                  # whether routing is uncertain
    fallback_free: bool              # whether free adapter should be used
    fallback_reason: str = ""        # why fallback: radius|subR2|conflict|uncertain|score|none
    # Score decomposition for debugging
    geo_score: float = 0.0           # -d虏 / temperature
    radius_ok: bool = True           # within radius
    subR2: float = 0.0               # chart subspace_r2
    conflict: float = 0.0            # chart overlap_rate
    best_score: float = 0.0          # final score of selected chart


class ChartRouterV3:
    """Routes features to charts via diagonal Mahalanobis distance.

    Simplified design: no trainable parameters, pure distance-based routing.
    Uses chart mu/var for distance and chart radius for within-chart decisions.

    Args:
        temperature: softmax temperature for routing weights
        top_k: number of top charts to consider (2 for L9/L10, 1 for L11)
        margin_threshold: margin below which routing is "uncertain"
        entropy_threshold: entropy above which routing falls back to free
        score_threshold: minimum score to accept chart routing
        min_full_r2_for_routing: minimum chart full_r2 to allow routing
        beta_conflict: penalty for overlapping charts
    """

    def __init__(
        self,
        temperature=1.0,
        top_k=2,
        margin_threshold=0.5,
        entropy_threshold=1.5,
        score_threshold=-2.0,
        min_full_r2_for_routing=0.0,
        beta_conflict=0.0,
        beta_quality=2.0,
        beta_task=0.1,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold
        self.score_threshold = score_threshold
        self.min_full_r2_for_routing = min_full_r2_for_routing
        self.beta_conflict = beta_conflict
        self.beta_quality = beta_quality
        self.beta_task = beta_task

    def compute_pair_scores(self, h, charts, adapters_by_chart):
        """Compute (chart, adapter) pair scores for feature h.

        Args:
            h: [D] single feature vector
            charts: List[ChartStateV3]
            adapters_by_chart: dict {chart_id: list of adapter slots}

        Returns:
            scores: [P] pair scores
            pair_info: list of (chart_idx, adapter_idx, within, sub_r2, conflict)
        """
        scores = []
        pair_info = []

        for k, chart in enumerate(charts):
            d2 = chart.mahalanobis_d2(h.unsqueeze(0)).squeeze(0)  # scalar
            within = d2 <= chart.radius_d2
            geo_score = -d2 / self.temperature
            conflict = getattr(chart, 'overlap_rate', 0.0)
            sub_r2 = getattr(chart, 'subspace_r2', 0.0)

            adapters = adapters_by_chart.get(chart.chart_id, [chart.adapter])
            if adapters is None or (isinstance(adapters, list) and len(adapters) == 0):
                adapters = []

            for aid, ad in enumerate(adapters):
                if ad is None:
                    continue
                # Key distance in adapter latent space
                if hasattr(ad, 'P') and hasattr(ad, 'key_adapt'):
                    chart_h = chart.transform_features(h)
                    z_adapt = torch.matmul(chart_h - chart.mu, ad.P)
                    key_d2 = ((z_adapt - ad.key_adapt) ** 2).sum()
                else:
                    key_d2 = 0.0

                pair_score = (
                    geo_score
                    - key_d2 / 2.0
                    - self.beta_conflict * conflict
                )
                scores.append(pair_score)
                pair_info.append((k, aid, within.item(), sub_r2, conflict))

        if len(scores) == 0:
            return torch.tensor([]), []

        return torch.stack(scores), pair_info

    def route_single(self, h, charts, adapters_by_chart=None):
        """Route a single feature vector to the best (chart, adapter) pair.

        Args:
            h: [D] feature vector
            charts: List[ChartStateV3]
            adapters_by_chart: optional dict of {chart_id: [adapters]}

        Returns:
            RouteResult
        """
        if adapters_by_chart is None:
            adapters_by_chart = {
                c.chart_id: list(c._adapters.values()) if c._adapters else []
                for c in charts
            }

        if len(charts) == 0:
            return RouteResult(
                selected_chart=None, selected_adapter=None,
                top1_distance=float('inf'), top2_distance=float('inf'),
                margin=float('inf'), entropy=0.0, weights=torch.zeros(0),
                uncertain=True, fallback_free=True, fallback_reason="no_charts",
            )

        # Compute distances to all charts + adapter quality bonus
        d2_values = []
        quality_bonus = []
        for chart in charts:
            d2 = chart.mahalanobis_d2(h.unsqueeze(0)).squeeze(0).item()
            d2_values.append(d2)
            # Adapter quality: higher subR2 = better adapter fit
            sub_r2 = getattr(chart, 'subspace_r2', 0.0)
            quality_bonus.append(self.beta_quality * sub_r2)

        d2_tensor = torch.tensor(d2_values, device=h.device)
        quality_tensor = torch.tensor(quality_bonus, device=h.device)

        # Adapter-aware score: geometry + quality (+ task bonus for recent charts)
        logits = -d2_tensor / self.temperature + quality_tensor

        # Sort by combined score
        sorted_scores, sorted_idx = logits.sort(descending=True)

        top1_distance = d2_tensor[sorted_idx[0]].item()
        top2_distance = d2_tensor[sorted_idx[1]].item() if len(sorted_idx) > 1 else float('inf')
        margin = top2_distance - top1_distance

        weights = torch.softmax(logits, dim=0)
        weights_clamped = weights.clamp_min(1e-8)
        entropy = -(weights_clamped * weights_clamped.log()).sum().item()

        # Uncertainty
        uncertain = margin < self.margin_threshold or entropy > self.entropy_threshold

        # Check if best chart is within radius and passes quality checks
        best_chart = charts[sorted_idx[0].item()]
        best_geo_score = logits[sorted_idx[0]].item()  # -d虏/T
        best_within = top1_distance <= (best_chart.radius_d2.item() if best_chart.radius_d2 is not None else float('inf'))
        best_sub_r2 = getattr(best_chart, 'subspace_r2', 0.0)
        best_full_r2 = getattr(best_chart, 'full_r2', 0.0)
        best_conflict = getattr(best_chart, 'overlap_rate', 0.0)

        # Hard fallback: only when geometry or adapter is truly unavailable.
        # Score is for ranking, NOT for hard rejection.
        fallback_free = False
        fallback_reason = "none"
        if len(charts) == 0:
            fallback_free = True; fallback_reason = "no_charts"
        elif not best_within:
            fallback_free = True; fallback_reason = "radius"
        elif best_chart.adapter is None:
            fallback_free = True; fallback_reason = "no_adapter"
        elif best_conflict >= 0.8:
            fallback_free = True; fallback_reason = "conflict"

        if fallback_free:
            return RouteResult(
                selected_chart=None, selected_adapter=None,
                top1_distance=top1_distance, top2_distance=top2_distance,
                margin=margin, entropy=entropy, weights=weights,
                uncertain=uncertain, fallback_free=True,
                fallback_reason=fallback_reason,
                geo_score=best_geo_score, radius_ok=best_within,
                subR2=best_sub_r2, conflict=best_conflict,
                best_score=logits.max().item(),
            )

        # Select best adapter by key distance (not just latest)
        best_adapter_id = 0
        if best_chart._adapters:
            best_key_d2 = float('inf')
            for tid, ad in best_chart._adapters.items():
                if hasattr(ad, 'P') and hasattr(ad, 'key_adapt'):
                    ad.to(h.device)  # adapter is separate module, must move explicitly
                    z = torch.matmul(h - best_chart.mu, ad.P)
                    kd2 = ((z - ad.key_adapt) ** 2).sum().item()
                else:
                    kd2 = 0.0
                if kd2 < best_key_d2:
                    best_key_d2 = kd2
                    best_adapter_id = tid

        return RouteResult(
            selected_chart=best_chart.chart_id,
            selected_adapter=best_adapter_id,
            top1_distance=top1_distance,
            top2_distance=top2_distance,
            margin=margin,
            entropy=entropy,
            weights=weights,
            uncertain=uncertain,
            fallback_free=False,
        )

    def route_batch(self, h, charts, adapters_by_chart=None):
        """Route a batch [B, D] of features.

        Returns:
            List[RouteResult] of length B
        """
        B = h.shape[0]
        return [self.route_single(h[b], charts, adapters_by_chart) for b in range(B)]

    def compute_routing_metrics(self, results):
        """Compute aggregate metrics from a list of RouteResults.

        Returns dict matching design doc Section 6.4.
        """
        if not results:
            return {
                "route_entropy": 0.0, "route_margin": 0.0,
                "top1_distance_mean": 0.0, "top2_distance_mean": 0.0,
                "uncertain_ratio": 0.0, "switch_rate": 0.0,
                "free_fallback_ratio": 0.0, "top1_chart_histogram": {},
                "num_routed": 0, "num_free_fallback": 0,
            }

        entropies = [r.entropy for r in results]
        margins = [r.margin for r in results if r.margin < float('inf')]
        top1s = [r.top1_distance for r in results if r.top1_distance < float('inf')]
        top2s = [r.top2_distance for r in results if r.top2_distance < float('inf')]
        uncertain_count = sum(1 for r in results if r.uncertain)
        free_count = sum(1 for r in results if r.fallback_free)
        chart_counts = {}
        fallback_reasons = {}
        for r in results:
            if r.selected_chart is not None:
                chart_counts[r.selected_chart] = chart_counts.get(r.selected_chart, 0) + 1
            if r.fallback_free:
                reason = r.fallback_reason or "unknown"
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

        total = len(results)
        return {
            "route_entropy": sum(entropies) / total if total > 0 else 0,
            "route_margin": sum(margins) / len(margins) if margins else float('inf'),
            "top1_distance_mean": sum(top1s) / len(top1s) if top1s else 0,
            "top2_distance_mean": sum(top2s) / len(top2s) if top2s else 0,
            "uncertain_ratio": uncertain_count / total if total > 0 else 0,
            "free_fallback_ratio": free_count / total if total > 0 else 0,
            "top1_chart_histogram": chart_counts,
            "fallback_reasons": fallback_reasons,
            "num_routed": total - free_count,
            "num_free_fallback": free_count,
        }


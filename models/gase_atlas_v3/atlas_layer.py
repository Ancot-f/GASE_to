"""GASE-Atlas v3 adaptation layer.

The layer owns adapters and chart state; routing decisions live in
``routing_policy.py`` so train-time free selection and inference share rules.
"""

import torch
import torch.nn as nn

from models.gase_atlas_v3.adapters import FreeAdapter, TaskAdapter
from models.gase_atlas_v3.chart_state import LayerAtlasState
from models.gase_atlas_v3.routing_policy import GASEAtlasRoutingPolicy


class GASEAtlasLayerV3(nn.Module):
    def __init__(
        self,
        layer_id,
        dim=768,
        task_bottleneck=16,
        free_bottleneck=16,
        routing_temperature=1.0,
        top_k=1,
        min_full_r2_for_routing=0.25,
        routing_beta_r2=3.0,
        routing_beta_conflict=0.5,
        min_adapter_cos=0.20,
        max_adapter_norm_ratio=2.0,
        uncertainty_margin=0.5,
        uncertainty_entropy=1.2,
        l11_disable_chart=True,
        l11_use_free=False,
        is_chart_domain=True,
        freeze_early_after_task0=True,
    ):
        super().__init__()
        self.layer_id = int(layer_id)
        self.dim = int(dim)
        self.is_chart_domain = bool(is_chart_domain)
        self.freeze_early_after_task0 = bool(freeze_early_after_task0)
        self.routing_temperature = float(routing_temperature)
        self.top_k = int(top_k)

        self.policy = GASEAtlasRoutingPolicy(
            temperature=routing_temperature,
            min_r2=min_full_r2_for_routing,
            min_adapter_cos=min_adapter_cos,
            max_norm_ratio=max_adapter_norm_ratio,
            uncertainty_margin=uncertainty_margin,
            uncertainty_entropy=uncertainty_entropy,
            beta_r2=routing_beta_r2,
            beta_conflict=routing_beta_conflict,
            l11_disable_chart=l11_disable_chart,
            l11_use_free=l11_use_free,
        )

        self._mode = "inference"
        self._current_task = 0
        self.task_adapter = TaskAdapter(dim=dim, bottleneck=task_bottleneck) if is_chart_domain else None
        self.free_adapter = FreeAdapter(dim=dim, bottleneck=free_bottleneck)
        self.atlas = LayerAtlasState(self.layer_id)
        self.chart_adapters = nn.ModuleList()
        self._teacher_flow_cache = None
        self._collect_enabled = False

    def set_mode(self, mode, current_task=None):
        self._mode = mode
        if current_task is not None:
            self._current_task = int(current_task)
        if mode == "task_train":
            self._prepare_task_train()
        elif mode == "inference":
            self._prepare_inference()

    def _prepare_task_train(self):
        if self.task_adapter is not None:
            for p in self.task_adapter.parameters():
                p.requires_grad = True
        for p in self.free_adapter.parameters():
            p.requires_grad = False

    def _prepare_inference(self):
        for p in self.parameters():
            p.requires_grad = False

    def reset_task_adapter(self):
        if self.task_adapter is not None:
            self.task_adapter.reset_parameters()
        self._teacher_flow_cache = None

    def remove_task_adapter_grads(self):
        if self.task_adapter is not None:
            for p in self.task_adapter.parameters():
                p.requires_grad = False

    def init_teacher_flow_cache(self, cache):
        self._teacher_flow_cache = cache

    def set_collect_enabled(self, enabled):
        self._collect_enabled = bool(enabled)

    def forward(self, x):
        if not self.is_chart_domain:
            return {"out": x + self.free_adapter(x), "records": {}, "route_results": []}
        if getattr(self, "l11_identity", False) and self.layer_id == 11:
            return {"out": x, "records": {}, "route_results": []}
        if self._mode == "task_train":
            return self._forward_task_train(x)
        return self._forward_inference(x)

    def _forward_task_train(self, x):
        delta = self.task_adapter(x)
        out = x + delta
        if self._collect_enabled and self._teacher_flow_cache is not None:
            self._teacher_flow_cache.record(
                self.layer_id,
                h_pre=x[:, 0].detach().cpu(),
                delta_task=delta[:, 0].detach().cpu(),
                h_post=out[:, 0].detach().cpu(),
            )
        return {"out": out, "records": {}, "route_results": []}

    def _forward_inference(self, x):
        B = x.shape[0]
        h = x[:, 0]
        delta = torch.zeros_like(x)
        free_delta = None
        route_results = []

        for i in range(B):
            decision = self.policy.decide(
                h[i:i + 1],
                self.atlas.active_charts(),
                self.layer_id,
                top_k=self.top_k,
            )
            route_results.append(decision)
            if decision.use_chart and decision.candidates:
                chart_delta, _, _ = self._mixed_chart_delta(
                    x[i:i + 1],
                    decision.candidates,
                    decision.weights,
                )
                delta[i] = self._cap_delta_norm(chart_delta, x[i:i + 1])[0]
            elif decision.use_free:
                if free_delta is None:
                    free_delta = self.free_adapter(x)
                delta[i] = self._cap_delta_norm(free_delta[i:i + 1], x[i:i + 1])[0]
            else:
                delta[i].zero_()

        return {"out": x + delta, "records": {}, "route_results": route_results}

    def compute_inference_delta_features(self, features):
        """Inference-equivalent delta for CLS features used by diagnostics."""
        delta = torch.zeros_like(features)
        free_delta = None
        charts = self.atlas.active_charts()
        for i in range(features.shape[0]):
            decision = self.policy.decide(
                features[i:i + 1],
                charts,
                self.layer_id,
                top_k=self.top_k,
            )
            if decision.use_chart and decision.candidates:
                chart_delta, quality, radius_ratio = self._mixed_chart_delta(
                    features[i:i + 1],
                    decision.candidates,
                    decision.weights,
                )
                chart_delta = self._cap_delta_norm(chart_delta, features[i:i + 1])[0]
                gamma = self._free_gamma(quality, radius_ratio)
                if gamma > 0.0 and free_delta is None:
                    free_delta = self.free_adapter(features)
                if gamma > 0.0:
                    mixed_delta = chart_delta.unsqueeze(0) + gamma * free_delta[i:i + 1]
                    delta[i] = self._cap_delta_norm(mixed_delta, features[i:i + 1])[0]
                else:
                    delta[i] = chart_delta
            elif decision.use_free:
                if free_delta is None:
                    free_delta = self.free_adapter(features)
                delta[i] = self._cap_delta_norm(free_delta[i:i + 1], features[i:i + 1])[0]
        return delta

    def _mixed_chart_delta(self, x, candidates, weights):
        if weights is None or weights.numel() != len(candidates):
            weights = x.new_ones(len(candidates)) / max(len(candidates), 1)

        mixed = torch.zeros_like(x)
        quality = 0.0
        radius_ratio = 0.0
        for weight, cand in zip(weights, candidates):
            if float(weight.item()) < 0.01:
                continue
            chart = cand.chart
            adapter = cand.adapter.to(x.device)
            part = adapter(x, chart)
            w = float(weight.item())
            mixed = mixed + w * part
            quality += w * max(cand.full_r2, cand.subspace_r2)
            radius_ratio += w * cand.radius_ratio
        return mixed, quality, radius_ratio

    def _free_gamma(self, quality, radius_ratio):
        quality = float(quality)
        radius_ratio = float(radius_ratio)
        if quality >= 0.65 and radius_ratio <= 1.0:
            return 0.0
        gamma = max(0.0, min(0.5, (0.65 - quality) / 0.65 * 0.5))
        if radius_ratio > 1.0:
            gamma = min(0.8, gamma + 0.2 * (radius_ratio - 1.0))
        return gamma

    def _cap_delta_norm(self, delta, ref, max_ratio=2.0):
        ref_norm = ref.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(max_ratio * ref_norm / delta_norm, max=1.0)
        return delta * scale

    def compute_chart_delta_features(self, features, only_charts=None):
        """Radius-based chart delta for free adapter training target.

        By default uses ALL active charts.  Pass *only_charts* (newly built
        charts) to match V2: old-chart adapter outputs are left untouched so
        the free adapter does not learn to compensate for them.
        """
        charts = only_charts if only_charts is not None else self.atlas.active_charts()
        delta = torch.zeros_like(features)
        if not charts:
            return delta
        for chart in charts:
            ad = chart.adapter
            if ad is None:
                continue
            chart.to(features.device)
            ad.to(features.device)
            in_chart = chart.within_radius(features)
            if in_chart.sum() > 0:
                with torch.no_grad():
                    pred = ad(features[in_chart], chart)
                delta[in_chart] = pred.detach()
            chart.cpu()
            ad.cpu()
        return delta

    def compute_free_mask(self, features):
        return self.policy.free_mask(features, self.atlas.active_charts(), self.layer_id)

    def register_charts(self, charts):
        for chart in charts:
            chart.layer_id = self.layer_id
            name = f"chart_{chart.chart_id}"
            suffix = 0
            while hasattr(self, name):
                suffix += 1
                name = f"chart_{chart.chart_id}_{suffix}"
            setattr(self, name, chart)
        self.atlas.register_charts(charts)

    @property
    def num_charts(self):
        return self.atlas.num_active

    @property
    def mode(self):
        return self._mode

"""GASE-Atlas v3 block module -thin wrapper integrating GASEAtlasLayerV3 into ViT Block.

Completely independent of V1. Uses GASEAtlasLayerV3 directly.
"""

import torch
import torch.nn as nn

from models.gase_atlas_v3.atlas_layer import GASEAtlasLayerV3


class GASEAtlasBlockModuleV3(nn.Module):
    """Adapter module inserted into ViT blocks.

    Layers 0-8:  FreeAdapter only (no charts, no task adapter)
    Layers 9-11: Full atlas -TaskAdapter (temp) + Charts + FreeAdapter
    """

    def __init__(self, config, layer_id, writer):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.writer = writer

        adapt_start = int(self._cfg("gase_atlas_adapt_start_layer", 9))
        adapt_end = int(self._cfg("gase_atlas_adapt_end_layer", 11))
        self.not_addition_layer = layer_id < adapt_start or layer_id > adapt_end

        d_model = int(self._cfg("d_model", 768))
        task_bottleneck = int(self._cfg("gase_atlas_task_bottleneck", 16))
        free_bottleneck = int(self._cfg("gase_atlas_free_bottleneck", 16))
        routing_temperature = float(self._cfg("gase_atlas_routing_temperature", 0.5))
        top_k = int(self._cfg("gase_atlas_top_k", 1))
        min_full_r2 = float(self._cfg("chart_min_full_R2_for_routing", 0.0))
        beta_r2 = float(self._cfg("routing_beta_r2", 0.0))
        beta_conflict = float(self._cfg("routing_beta_conflict", 0.0))
        min_adapter_cos = float(self._cfg("routing_min_adapter_cos", 0.20))
        max_adapter_norm_ratio = float(self._cfg("routing_max_adapter_norm_ratio", 2.0))
        uncertainty_margin = float(self._cfg("routing_uncertainty_margin", 0.5))
        uncertainty_entropy = float(self._cfg("routing_uncertainty_entropy", 1.2))
        l11_disable_chart = bool(self._cfg("gase_atlas_l11_disable_chart", True))
        l11_use_free = bool(self._cfg("gase_atlas_l11_use_free", False))
        freeze_early = bool(self._cfg("gase_atlas_freeze_early_after_task0", True))

        self.atlas_layer = GASEAtlasLayerV3(
            layer_id=layer_id,
            dim=d_model,
            task_bottleneck=task_bottleneck if not self.not_addition_layer else None,
            free_bottleneck=free_bottleneck,
            routing_temperature=routing_temperature,
            top_k=top_k,
            min_full_r2_for_routing=min_full_r2,
            routing_beta_r2=beta_r2,
            routing_beta_conflict=beta_conflict,
            min_adapter_cos=min_adapter_cos,
            max_adapter_norm_ratio=max_adapter_norm_ratio,
            uncertainty_margin=uncertainty_margin,
            uncertainty_entropy=uncertainty_entropy,
            l11_disable_chart=l11_disable_chart,
            l11_use_free=l11_use_free,
            is_chart_domain=not self.not_addition_layer,
            freeze_early_after_task0=freeze_early,
        )
        if layer_id == 11:
            self.atlas_layer.l11_chart_scale = float(self._cfg("gase_atlas_l11_chart_scale", 1.0))
            self.atlas_layer.l11_identity = bool(self._cfg("gase_atlas_l11_identity", False))

        self._pending_records = []
        self._no_adapter = False

    def _cfg(self, name, default):
        return getattr(self.config, name, default)

    # ---- Forward ----

    def forward(self, x):
        if self._no_adapter:
            delta = torch.zeros_like(x)
            B = x.shape[0]
            return {
                "func_out": delta,
                "rd_loss": torch.tensor(0.0, device=x.device),
                "added": False,
                "router_entropy": torch.zeros(B, device=x.device),
                "rd_score": torch.zeros(B, device=x.device),
                "coverage_score": torch.ones(B, device=x.device),
                "expert_router_entropy": torch.zeros(B, device=x.device),
                "chart_weights": x.new_zeros(B, 0),
                "chart_coverages": x.new_zeros(B, 0),
                "chart_accepted": torch.zeros(B, device=x.device, dtype=torch.bool),
                "selected_charts": torch.full((B, 0), -1, device=x.device, dtype=torch.long),
                "selected_adapters": torch.full((B, 0), -1, device=x.device, dtype=torch.long),
                "adapter_key_loss": x.new_zeros(()),
                "adapter_sep_loss": x.new_zeros(()),
                "task_distill_loss": x.new_zeros(()),
                "free_fallback_mask": torch.zeros(B, device=x.device, dtype=torch.bool),
                "num_charts": 0, "num_chart_adapters": 0,
                "chart_adapter_layout": "[]",
            }
        result = self.atlas_layer(x)
        delta = result["out"] - x
        B = x.shape[0]
        return {
            "func_out": delta,
            "rd_loss": torch.tensor(0.0, device=x.device),
            "added": False,
            "router_entropy": torch.zeros(B, device=x.device),
            "rd_score": torch.zeros(B, device=x.device),
            "coverage_score": torch.ones(B, device=x.device),
            "expert_router_entropy": torch.zeros(B, device=x.device),
            "chart_weights": x.new_zeros(B, 0),
            "chart_coverages": x.new_zeros(B, 0),
            "chart_accepted": torch.zeros(B, device=x.device, dtype=torch.bool),
            "selected_charts": torch.full((B, 0), -1, device=x.device, dtype=torch.long),
            "selected_adapters": torch.full((B, 0), -1, device=x.device, dtype=torch.long),
            "adapter_key_loss": x.new_zeros(()),
            "adapter_sep_loss": x.new_zeros(()),
            "task_distill_loss": x.new_zeros(()),
            "free_fallback_mask": torch.zeros(B, device=x.device, dtype=torch.bool),
            "num_charts": self.atlas_layer.num_charts,
            "num_chart_adapters": self._num_chart_adapters(),
            "chart_adapter_layout": self._layout_str(),
        }

    # ---- Phase / mode control ----

    def set_train_phase(self, current_task=None):
        if current_task is not None:
            self.atlas_layer.set_mode("task_train", int(current_task))
        if self.not_addition_layer:
            freeze = (current_task is not None and int(current_task) > 0
                      and self.atlas_layer.freeze_early_after_task0)
            for p in self.atlas_layer.free_adapter.parameters():
                p.requires_grad = not freeze
        else:
            for p in self.atlas_layer.task_adapter.parameters():
                p.requires_grad = True
            for p in self.atlas_layer.free_adapter.parameters():
                p.requires_grad = False

    def set_eval_phase(self):
        for p in self.parameters():
            p.requires_grad = False

    # ---- End of task ----

    def end_of_task(self, age=0):
        if self.not_addition_layer:
            return {"charts_created": 0, "charts_mapped": 0,
                    "charts_merged": 0, "adapters_added": 0}
        result = {"charts_created": 0, "charts_mapped": 0,
                  "charts_merged": 0, "adapters_added": 0}
        self.atlas_layer.set_mode("inference")
        self.atlas_layer.remove_task_adapter_grads()
        return result

    def flush_pending_records(self):
        records = self._pending_records
        self._pending_records = []
        return records

    def _layout_str(self):
        if self.not_addition_layer:
            return "[free]"
        slots = [f"{c.chart_id}:{c.num_adapters}" for c in self.atlas_layer.atlas.charts]
        return f"[charts={self.atlas_layer.num_charts} slots={','.join(slots)}]"

    def _num_chart_adapters(self):
        if self.not_addition_layer:
            return 0
        return sum(c.num_adapters for c in self.atlas_layer.atlas.charts)

    @property
    def num_charts(self):
        return self.atlas_layer.num_charts

    def log_selector_entropy(self):
        pass


# Alias for compatibility
GASEAtlasFiberUnitV3 = GASEAtlasBlockModuleV3


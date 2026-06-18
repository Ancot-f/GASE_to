"""
GASEAtlasBlock: ViT transformer block augmented with GASE adapters.

Phase-6.5: L0-L11 all wrapped as GASEAtlasBlock.
  L0-L8: base_adapter (frozen after Task0), no chart/slot
  L9-L11: task_adapter → chart/slot/chart_adapter/free_adapter
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from .gase_components import (
    TASK_TRAIN, TASK0_BOOTSTRAP, BASE_PLUS_TASK_TRAIN,
    DISTILL, INFER,
    L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT,
    CURRENT_SLOT_STUDENT, ORACLE_SLOT_STUDENT, KEY_SLOT_STUDENT,
    PATH_KEY_SLOT_STUDENT,
    RoutingOutput,
)

_VALID_MODES = (
    TASK_TRAIN, TASK0_BOOTSTRAP, BASE_PLUS_TASK_TRAIN,
    DISTILL, INFER,
    L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT,
    CURRENT_SLOT_STUDENT, ORACLE_SLOT_STUDENT, KEY_SLOT_STUDENT,
    PATH_KEY_SLOT_STUDENT,
)


class GASEAtlasBlock(nn.Module):
    """A ViT block augmented with GASE atlas + multi-slot + base_adapter."""

    def __init__(self, original_block: nn.Module, layer_id: int, dim: int, config: dict):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.original_block = original_block

        # Layer classification
        base_layers = config.get("base_adapter_layers", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        atlas_layers = config.get("atlas_layers", [9, 10, 11])
        self.is_base_layer: bool = layer_id in base_layers
        self.is_atlas_layer: bool = layer_id in atlas_layers

        # Adapters
        self.base_adapter: Optional[nn.Module] = None
        self.task_adapter: Optional[nn.Module] = None
        self.chart_router: Optional[nn.Module] = None
        self.slot_router: Optional[nn.Module] = None
        self.chart_adapters: Dict[str, nn.Module] = nn.ModuleDict()
        self.free_adapters: Dict[str, nn.Module] = nn.ModuleDict()
        self.chart_states: Dict[int, object] = {}
        self.slot_states: Dict[str, object] = {}

        self.adapter_mode: str = INFER
        self.active_slot_id: Optional[int] = None
        self.oracle_slot_id: Optional[int] = None
        self.path_slot_id: Optional[Tensor] = None  # [B] for PATH_KEY_SLOT_STUDENT
        self.last_routing_info: Optional[Dict[str, Any]] = None
        self.last_path_routing_info: Optional[Dict[str, Any]] = None

        routing_cfg = config.get("routing", {})
        self.use_free_adapter: bool = config.get("free_adapter", {}).get("enabled", True)
        self.use_soft_chart_routing: bool = routing_cfg.get("use_soft_chart_routing", True)
        self.use_identity_fallback: bool = routing_cfg.get("use_identity_fallback", True)

    # ==================================================================
    #  Forward
    # ==================================================================

    def forward(self, x: Tensor, return_routing: bool = False) -> Tuple[Tensor, Optional[RoutingOutput]]:
        x = self.forward_original_block(x)
        h_chart = self.get_router_feature(x)
        routing_info = None

        mode = self.adapter_mode

        # --- TASK0_BOOTSTRAP: L0-L11 all use task_adapter ---
        if mode == TASK0_BOOTSTRAP:
            if self.task_adapter is not None:
                delta = self.apply_task_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)

        # --- BASE_PLUS_TASK_TRAIN: L0-L8 base, L9-L11 task_adapter ---
        elif mode == BASE_PLUS_TASK_TRAIN:
            if self.is_base_layer and self.base_adapter is not None:
                delta = self.apply_base_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.is_atlas_layer and self.task_adapter is not None:
                delta = self.apply_task_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)

        # --- TASK_TRAIN (legacy): task_adapter on atlas layers ---
        elif mode == TASK_TRAIN:
            if self.task_adapter is not None:
                delta = self.apply_task_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)

        # --- CURRENT_SLOT_STUDENT: base + committed slot or task ---
        elif mode == CURRENT_SLOT_STUDENT:
            if self.is_base_layer and self.base_adapter is not None:
                delta = self.apply_base_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.is_atlas_layer:
                sid = self.active_slot_id
                if sid is not None and self._has_slot_adapter(sid):
                    delta = self._apply_chart_free_combined(h_chart, sid)
                    x = self.add_delta_to_cls(x, delta)
                elif self.task_adapter is not None:
                    delta = self.apply_task_adapter(h_chart)
                    x = self.add_delta_to_cls(x, delta)

        # --- ORACLE_SLOT_STUDENT: base + oracle slot ---
        elif mode == ORACLE_SLOT_STUDENT:
            if self.is_base_layer and self.base_adapter is not None:
                delta = self.apply_base_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.is_atlas_layer:
                sid = self.oracle_slot_id
                if sid is not None:
                    delta = self._apply_chart_free_combined(h_chart, sid)
                    x = self.add_delta_to_cls(x, delta)

        # --- KEY_SLOT_STUDENT: per-sample key-selected slot ---
        elif mode == KEY_SLOT_STUDENT:
            if self.is_base_layer and self.base_adapter is not None:
                delta = self.apply_base_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.is_atlas_layer:
                routing = self.select_slots_per_sample_by_key(h_chart, chart_id=0)
                selected = routing["slot_ids"]
                delta = self.apply_chart_adapters_per_sample(h_chart, selected, chart_id=0)
                self.last_routing_info = routing
                x = self.add_delta_to_cls(x, delta)

        # --- PATH_KEY_SLOT_STUDENT: L9 decides path, L10/L11 follow ---
        elif mode == PATH_KEY_SLOT_STUDENT:
            if self.is_base_layer and self.base_adapter is not None:
                delta = self.apply_base_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.is_atlas_layer:
                if self.path_slot_id is not None:
                    # Follow L9-decided path
                    delta = self.apply_chart_adapters_per_sample(h_chart, self.path_slot_id, chart_id=0)
                    x = self.add_delta_to_cls(x, delta)
                else:
                    # L9: decide the path
                    routing = self.select_slots_per_sample_by_key(h_chart, chart_id=0)
                    self.last_routing_info = routing
                    self.last_path_routing_info = routing
                    self.path_slot_id = routing["slot_ids"]
                    delta = self.apply_chart_adapters_per_sample(h_chart, self.path_slot_id, chart_id=0)
                    x = self.add_delta_to_cls(x, delta)

        # --- SEQUENTIAL_CHART_STUDENT / L9_CHART_STUDENT (legacy) ---
        elif mode in (L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT):
            if self.has_active_chart_adapter():
                delta = self.apply_first_chart_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.task_adapter is not None:
                delta = self.apply_task_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)

        return x, routing_info

    def _has_slot_adapter(self, slot_id: int) -> bool:
        return f"0_{slot_id}" in self.chart_adapters

    def _apply_chart_free_combined(self, h_chart: Tensor, slot_id: int) -> Tensor:
        """Combine chart-adapter + free-adapter for a slot."""
        delta_chart = self.apply_chart_adapter_by_slot(h_chart, chart_id=0, slot_id=slot_id)
        delta_free = self.apply_free_adapter_by_slot(h_chart, slot_id)
        return delta_chart + delta_free

    # ==================================================================
    #  Core helpers
    # ==================================================================

    def forward_original_block(self, x: Tensor) -> Tensor:
        return self.original_block(x)

    def add_delta_to_cls(self, x: Tensor, delta: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + delta
            return x
        elif x.dim() == 2:
            return x + delta
        raise ValueError(f"Expected [B,N,D] or [B,D], got {x.shape}")

    def get_router_feature(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            return x[:, 0]
        elif x.dim() == 2:
            return x
        raise ValueError(f"Expected [B,N,D] or [B,D], got {x.shape}")

    # ==================================================================
    #  Base adapter
    # ==================================================================

    def set_base_adapter(self, adapter: nn.Module, freeze: bool = True) -> None:
        self.base_adapter = adapter
        if freeze:
            for p in self.base_adapter.parameters():
                p.requires_grad = False

    def has_base_adapter(self) -> bool:
        return self.base_adapter is not None

    def apply_base_adapter(self, h_chart: Tensor) -> Tensor:
        if self.base_adapter is None:
            return torch.zeros_like(h_chart)
        return self.base_adapter(h_chart)

    def freeze_base_adapter(self) -> None:
        if self.base_adapter is not None:
            for p in self.base_adapter.parameters():
                p.requires_grad = False

    # ==================================================================
    #  Adapter application
    # ==================================================================

    def apply_task_adapter(self, h_chart: Tensor) -> Tensor:
        if self.task_adapter is None:
            return torch.zeros_like(h_chart)
        return self.task_adapter(h_chart)

    def has_active_chart_adapter(self) -> bool:
        if len(self.chart_adapters) == 0 or len(self.chart_states) == 0:
            return False
        cs = self.chart_states.get(0)
        return cs is not None and getattr(cs, "mu", None) is not None

    def apply_first_chart_adapter(self, h_chart: Tensor) -> Tensor:
        if not self.has_active_chart_adapter():
            return torch.zeros_like(h_chart)
        first_key = next(iter(self.chart_adapters))
        adapter = self.chart_adapters[first_key]
        cs = self.chart_states.get(0)
        mu = cs.mu.to(h_chart.device)
        return adapter(h_chart, mu)

    def apply_chart_adapter_by_slot(self, h_chart: Tensor, chart_id: int, slot_id: int) -> Tensor:
        key = f"{chart_id}_{slot_id}"
        if key not in self.chart_adapters:
            return torch.zeros_like(h_chart)
        adapter = self.chart_adapters[key]
        cs = self.chart_states.get(chart_id)
        if cs is None or getattr(cs, "mu", None) is None:
            return torch.zeros_like(h_chart)
        mu = cs.mu.to(h_chart.device)
        return adapter(h_chart, mu)

    def apply_free_adapter_by_slot(self, h_chart: Tensor, slot_id: int) -> Tensor:
        key = f"free_{slot_id}"
        if key in self.free_adapters:
            return self.free_adapters[key](h_chart)
        return torch.zeros_like(h_chart)

    # ==================================================================
    #  Slot selection
    # ==================================================================

    def select_slot_by_key(self, h_chart: Tensor, chart_id: int = 0) -> Optional[Tensor]:
        available_slots = self.get_available_slot_ids(chart_id)
        if not available_slots:
            return None
        keys, slot_ids_for_keys = [], []
        for sid in available_slots:
            sstate = self.slot_states.get(f"{chart_id}_{sid}")
            if sstate is not None and getattr(sstate, "key", None) is not None:
                keys.append(sstate.key.to(h_chart.device))
                slot_ids_for_keys.append(sid)
        if not keys:
            return None
        key_stack = torch.stack(keys)
        first_sstate = self.slot_states.get(f"{chart_id}_{available_slots[0]}")
        if first_sstate is None or getattr(first_sstate, "P", None) is None:
            return None
        P = first_sstate.P.to(h_chart.device)
        cs = self.chart_states.get(chart_id)
        if cs is not None and getattr(cs, "mu", None) is not None:
            h_proj = (h_chart - cs.mu.to(h_chart.device).unsqueeze(0)) @ P
        else:
            h_proj = h_chart @ P
        dists = torch.cdist(h_proj, key_stack)
        nearest = dists.argmin(dim=1)
        return torch.tensor([slot_ids_for_keys[i.item()] for i in nearest], device=h_chart.device)

    def get_available_slot_ids(self, chart_id: int = 0) -> List[int]:
        prefix = f"{chart_id}_"
        ids = []
        for key in self.chart_adapters:
            if key.startswith(prefix):
                try:
                    ids.append(int(key[len(prefix):]))
                except ValueError:
                    pass
        return sorted(ids)

    # ==================================================================
    #  Per-sample slot routing (Phase-7)
    # ==================================================================

    def select_slots_per_sample_by_key(
        self, h_chart: Tensor, chart_id: int = 0,
    ) -> Dict[str, Any]:
        """Select a slot for each sample independently using KeySlotRouter."""
        cs = self.chart_states.get(chart_id)
        if cs is None or getattr(cs, "mu", None) is None:
            return {"slot_ids": torch.zeros(h_chart.shape[0], dtype=torch.long, device=h_chart.device),
                    "entropy": torch.zeros(h_chart.shape[0], device=h_chart.device),
                    "margin": torch.full([h_chart.shape[0]], float("inf"), device=h_chart.device),
                    "slot_id_list": []}
        available = self.get_available_slot_ids(chart_id)
        slot_states: Dict[int, object] = {}
        for sid in available:
            ss = self.slot_states.get(f"{chart_id}_{sid}")
            if ss is not None and getattr(ss, "key", None) is not None:
                slot_states[sid] = ss
        if not slot_states:
            return {"slot_ids": torch.zeros(h_chart.shape[0], dtype=torch.long, device=h_chart.device),
                    "entropy": torch.zeros(h_chart.shape[0], device=h_chart.device),
                    "margin": torch.full([h_chart.shape[0]], float("inf"), device=h_chart.device),
                    "slot_id_list": []}
        from gase.routing.key_router import KeySlotRouter
        router = KeySlotRouter(temperature=1.0, use_mahalanobis=True)
        return router.route(h_chart, cs, slot_states)

    def apply_chart_adapters_per_sample(
        self, h_chart: Tensor, selected_slot_ids: Tensor, chart_id: int = 0,
    ) -> Tensor:
        """Apply different chart+free adapters per sample, scatter back to batch order."""
        assert selected_slot_ids.shape[0] == h_chart.shape[0]
        B, D = h_chart.shape
        delta = torch.zeros(B, D, device=h_chart.device, dtype=h_chart.dtype)
        for sid in selected_slot_ids.unique().tolist():
            mask = selected_slot_ids == sid
            if mask.sum() == 0:
                continue
            delta[mask] = (self.apply_chart_adapter_by_slot(h_chart[mask], chart_id, sid)
                           + self.apply_free_adapter_by_slot(h_chart[mask], sid))
        return delta

    # ==================================================================
    #  Slot id setters
    # ==================================================================

    def set_active_slot_id(self, slot_id: Optional[int]) -> None:
        self.active_slot_id = slot_id

    def set_oracle_slot_id(self, slot_id: Optional[int]) -> None:
        self.oracle_slot_id = slot_id

    # ==================================================================
    #  Registration
    # ==================================================================

    def set_adapter_mode(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown adapter_mode: {mode}")
        self.adapter_mode = mode

    def register_chart(self, chart_state: object) -> None:
        self.chart_states[chart_state.chart_id] = chart_state

    def register_slot(self, slot_state: object) -> None:
        key = f"{slot_state.chart_id}_{slot_state.slot_id}"
        self.slot_states[key] = slot_state

    def register_chart_adapter(
        self, chart_id: int, slot_id: int, adapter: nn.Module, freeze: bool = True,
    ) -> None:
        key = f"{chart_id}_{slot_id}"
        if key in self.chart_adapters:
            raise ValueError(
                f"Adapter already registered for chart={chart_id} slot={slot_id} "
                f"at layer {self.layer_id}."
            )
        self.chart_adapters[key] = adapter
        if freeze:
            for p in adapter.parameters():
                p.requires_grad = False

    def register_free_adapter(self, slot_id: int, adapter: nn.Module, freeze: bool = True) -> None:
        key = f"free_{slot_id}"
        self.free_adapters[key] = adapter
        if freeze:
            for p in adapter.parameters():
                p.requires_grad = False

    def remove_task_adapter(self) -> None:
        self.task_adapter = None

    def freeze_permanent_adapters(self) -> None:
        for adapter in self.chart_adapters.values():
            for p in adapter.parameters():
                p.requires_grad = False
        for adapter in self.free_adapters.values():
            for p in adapter.parameters():
                p.requires_grad = False

    def unfreeze_task_adapter(self) -> None:
        if self.task_adapter is not None:
            for p in self.task_adapter.parameters():
                p.requires_grad = True

    # ==================================================================
    #  Feature extraction (Phase-3 compat)
    # ==================================================================

    def extract_h_chart_and_delta_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        block_output = self.forward_original_block(x)
        h_chart = self.get_router_feature(block_output)
        delta_teacher = self.apply_task_adapter(h_chart)
        return block_output, h_chart, delta_teacher

    def apply_chart_adapters(self, h_chart, chart_probs=None, slot_probs=None):
        raise NotImplementedError("Use slot-based methods.")

    def combine_residuals(self, delta_task, delta_chart, delta_free, routing_info=None):
        raise NotImplementedError("Use per-mode residual logic in forward().")

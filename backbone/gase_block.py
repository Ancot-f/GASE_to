"""
GASEAtlasBlock: ViT transformer block augmented with GASE adapters.

Phase-6: supports multi-slot storage (one chart, many slots).
Each slot = one task. Old slots are frozen.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from .gase_components import (
    TASK_TRAIN, DISTILL, INFER,
    L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT,
    CURRENT_SLOT_STUDENT, ORACLE_SLOT_STUDENT, KEY_SLOT_STUDENT,
    RoutingOutput,
)

_VALID_MODES = (
    TASK_TRAIN, DISTILL, INFER,
    L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT,
    CURRENT_SLOT_STUDENT, ORACLE_SLOT_STUDENT, KEY_SLOT_STUDENT,
)


class GASEAtlasBlock(nn.Module):
    """A ViT block augmented with GASE atlas + multi-slot capabilities."""

    def __init__(self, original_block: nn.Module, layer_id: int, dim: int, config: dict):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.original_block = original_block

        self.task_adapter: Optional[nn.Module] = None
        self.chart_router: Optional[nn.Module] = None
        self.slot_router: Optional[nn.Module] = None
        self.chart_adapters: Dict[str, nn.Module] = nn.ModuleDict()
        self.chart_states: Dict[int, object] = {}       # chart_id -> ChartState
        self.slot_states: Dict[str, object] = {}         # "{chart_id}_{slot_id}" -> SlotState
        self.free_adapter: Optional[nn.Module] = None

        self.adapter_mode: str = INFER
        self.active_slot_id: Optional[int] = None
        self.oracle_slot_id: Optional[int] = None

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

        if self.adapter_mode == TASK_TRAIN and self.task_adapter is not None:
            delta = self.apply_task_adapter(h_chart)
            x = self.add_delta_to_cls(x, delta)

        elif self.adapter_mode in (L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT):
            if self.has_active_chart_adapter():
                delta = self.apply_first_chart_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.task_adapter is not None:
                delta = self.apply_task_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)

        elif self.adapter_mode == CURRENT_SLOT_STUDENT:
            if self.active_slot_id is not None:
                delta = self.apply_chart_adapter_by_slot(h_chart, chart_id=0, slot_id=self.active_slot_id)
                x = self.add_delta_to_cls(x, delta)
            elif self.has_active_chart_adapter():
                delta = self.apply_first_chart_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)
            elif self.task_adapter is not None:
                delta = self.apply_task_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta)

        elif self.adapter_mode == ORACLE_SLOT_STUDENT:
            if self.oracle_slot_id is not None:
                delta = self.apply_chart_adapter_by_slot(h_chart, chart_id=0, slot_id=self.oracle_slot_id)
                x = self.add_delta_to_cls(x, delta)
            # else identity

        elif self.adapter_mode == KEY_SLOT_STUDENT:
            slot_ids = self.select_slot_by_key(h_chart, chart_id=0)
            if slot_ids is not None:
                majority = int(torch.mode(slot_ids).values.item())
                delta = self.apply_chart_adapter_by_slot(h_chart, chart_id=0, slot_id=majority)
                x = self.add_delta_to_cls(x, delta)

        return x, routing_info

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
        chart_state = self.chart_states.get(0)
        mu = chart_state.mu.to(h_chart.device)
        return adapter(h_chart, mu)

    def apply_chart_adapter_by_slot(self, h_chart: Tensor, chart_id: int, slot_id: int) -> Tensor:
        """Apply a specific (chart, slot) adapter."""
        key = f"{chart_id}_{slot_id}"
        if key not in self.chart_adapters:
            return torch.zeros_like(h_chart)
        adapter = self.chart_adapters[key]
        cs = self.chart_states.get(chart_id)
        if cs is None or getattr(cs, "mu", None) is None:
            return torch.zeros_like(h_chart)
        mu = cs.mu.to(h_chart.device)
        return adapter(h_chart, mu)

    def apply_free_adapter(self, h_chart: Tensor) -> Tensor:
        if self.free_adapter is None:
            return torch.zeros_like(h_chart)
        return self.free_adapter(h_chart)

    # ==================================================================
    #  Slot selection
    # ==================================================================

    def select_slot_by_key(self, h_chart: Tensor, chart_id: int = 0) -> Optional[Tensor]:
        """Select nearest slot by L2 distance in P-space. Returns [B] slot_ids."""
        available_slots = self.get_available_slot_ids(chart_id)
        if not available_slots:
            return None
        # Collect keys for available slots
        keys = []
        slot_ids_for_keys = []
        for sid in available_slots:
            sstate = self.slot_states.get(f"{chart_id}_{sid}")
            if sstate is not None and getattr(sstate, "key", None) is not None:
                k = sstate.key.to(h_chart.device)  # [input_rank]
                keys.append(k)
                slot_ids_for_keys.append(sid)
        if not keys:
            return None
        key_stack = torch.stack(keys)  # [num_slots, input_rank]
        # Get P from first available slot state for projection
        first_sstate = self.slot_states.get(f"{chart_id}_{available_slots[0]}")
        if first_sstate is None or getattr(first_sstate, "P", None) is None:
            return None
        P = first_sstate.P.to(h_chart.device)  # [D, input_rank]
        # Center by chart mu
        cs = self.chart_states.get(chart_id)
        if cs is not None and getattr(cs, "mu", None) is not None:
            h_proj = (h_chart - cs.mu.to(h_chart.device).unsqueeze(0)) @ P  # [B, input_rank]
        else:
            h_proj = h_chart @ P
        # L2 distance to each key
        dists = torch.cdist(h_proj, key_stack)  # [B, num_slots]
        nearest = dists.argmin(dim=1)  # [B]
        return torch.tensor([slot_ids_for_keys[i.item()] for i in nearest], device=h_chart.device)

    def get_available_slot_ids(self, chart_id: int = 0) -> List[int]:
        """Return sorted slot ids registered under chart_id."""
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
        """Register a chart-slot adapter. Old slots are preserved (not overwritten)."""
        key = f"{chart_id}_{slot_id}"
        if key in self.chart_adapters:
            raise ValueError(
                f"Adapter already registered for chart={chart_id} slot={slot_id} "
                f"at layer {self.layer_id}. Old slots must not be overwritten."
            )
        self.chart_adapters[key] = adapter
        if freeze:
            for p in adapter.parameters():
                p.requires_grad = False

    def remove_task_adapter(self) -> None:
        self.task_adapter = None

    def freeze_permanent_adapters(self) -> None:
        for adapter in self.chart_adapters.values():
            for p in adapter.parameters():
                p.requires_grad = False
        if self.free_adapter is not None:
            for p in self.free_adapter.parameters():
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
        raise NotImplementedError("Phase-6 uses slot-based adapter application.")

    def combine_residuals(self, delta_task, delta_chart, delta_free, routing_info=None):
        raise NotImplementedError("Phase-6 uses per-mode residual logic in forward().")

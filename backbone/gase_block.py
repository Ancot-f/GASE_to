"""
GASEAtlasBlock: ViT transformer block augmented with GASE adapters.

Key modes:
- "task_train": uses task_adapter for teacher residual generation.
- "distill": collects h_chart (pre-current-adapter feature on permanent path)
  and teacher residual for distilling chart/free adapters.
- "infer": uses chart-adapter/free-adapter; task-adapter is disabled.

CRITICAL: h_chart is the feature that the router sees at inference time —
it is the pre-current-adapter feature on the permanent path, NOT the
task-adapter output, NOT backbone-fixed features.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from .gase_components import TASK_TRAIN, DISTILL, INFER, ResidualOutput, RoutingOutput


class GASEAtlasBlock(nn.Module):
    """
    A ViT block augmented with GASE atlas capabilities.

    Wraps an original ViT block and adds:
    - TaskAdapter (temporary, per-task teacher).
    - ChartRouter + ChartAdapters (permanent, task-agnostic).
    - SlotRouter (key-based or learned).
    - FreeAdapter (permanent, catches leftover residual).

    Attributes:
        layer_id: ViT block index (e.g., 9, 10, 11).
        dim: feature dimension D.
        original_block: the original ViT transformer block.
        task_adapter: temporary per-task adapter (TaskAdapter).
        chart_router: routes features to top-m charts.
        slot_router: routes features to top-k slots within a chart.
        chart_adapters: dict (chart_id, slot_id) -> ChartAdapter.
        free_adapter: FreeAdapter for residual leftover.
        adapter_mode: current mode (TASK_TRAIN / DISTILL / INFER).
        use_free_adapter: whether free-adapter is enabled.
        use_soft_chart_routing: whether to soft-mix multiple chart outputs.
        use_identity_fallback: whether to fall back to identity on high uncertainty.
    """

    def __init__(
        self,
        original_block: nn.Module,
        layer_id: int,
        dim: int,
        config: dict,
    ):
        """
        Args:
            original_block: the original ViT Block to wrap.
            layer_id: ViT block index.
            dim: feature dimension D.
            config: GASE configuration dict.
        """
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.original_block = original_block

        # Placeholder modules (created during first task)
        self.task_adapter: Optional[nn.Module] = None
        self.chart_router: Optional[nn.Module] = None
        self.slot_router: Optional[nn.Module] = None
        self.chart_adapters: Dict[Tuple[int, int], nn.Module] = nn.ModuleDict()
        self.free_adapter: Optional[nn.Module] = None

        self.adapter_mode: str = INFER

        routing_cfg = config.get("routing", {})
        self.use_free_adapter: bool = config.get("free_adapter", {}).get("enabled", True)
        self.use_soft_chart_routing: bool = routing_cfg.get("use_soft_chart_routing", True)
        self.use_identity_fallback: bool = routing_cfg.get("use_identity_fallback", True)

    def forward(
        self,
        x: Tensor,
        return_routing: bool = False,
    ) -> Tuple[Tensor, Optional[RoutingOutput]]:
        """
        Forward pass through the GASE block.

        Args:
            x: input features of shape [B, N, D].
            return_routing: if True, also return routing info.

        Returns:
            Tuple of (output [B, N, D], optional RoutingOutput).
        """
        x = self.forward_original_block(x)
        routing_info = None

        if self.adapter_mode == TASK_TRAIN and self.task_adapter is not None:
            h_chart = self.get_router_feature(x)
            delta_task = self.apply_task_adapter(h_chart)
            # Apply residual to CLS token (index 0)
            if x.dim() == 3 and x.shape[1] > 1 and delta_task.shape == x[:, 0].shape:
                x = x.clone()
                x[:, 0] = x[:, 0] + delta_task
        elif self.adapter_mode == DISTILL:
            # Placeholder: in real impl, collect h_chart and delta_teacher for caching.
            pass
        elif self.adapter_mode == INFER:
            # Phase-1: identity (chart/slot routing not yet implemented).
            pass

        return x, routing_info

    def forward_original_block(self, x: Tensor) -> Tensor:
        """
        Forward through the original ViT block (attention + MLP).

        Args:
            x: input features [B, N, D].

        Returns:
            Output features [B, N, D].
        """
        return self.original_block(x)

    def get_router_feature(self, x: Tensor) -> Tensor:
        """
        Extract the feature used for chart/slot routing.

        h_chart is the pre-current-adapter feature on the permanent path,
        computed before any adapter is applied at this block.

        Args:
            x: input features [B, N, D].

        Returns:
            h_chart of shape [B, D] (cls token).
        """
        if x.dim() == 3:
            return x[:, 0]  # CLS token
        return x  # already [B, D]

    def apply_task_adapter(self, h_chart: Tensor) -> Tensor:
        """
        Apply task-adapter in task_train mode.

        Args:
            h_chart: pre-adapter features [B, D].

        Returns:
            delta_task of shape [B, D].
        """
        if self.task_adapter is None:
            return torch.zeros_like(h_chart)
        return self.task_adapter(h_chart)

    def apply_chart_adapters(
        self,
        h_chart: Tensor,
        chart_probs: Optional[Tensor] = None,
        slot_probs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply chart-adapters with routing.

        Args:
            h_chart: pre-adapter features [B, D].
            chart_probs: chart probabilities [B, num_charts].
            slot_probs: slot probabilities [B, num_slots].

        Returns:
            delta_chart of shape [B, D].
        """
        # Phase-1 placeholder: return zeros. Real routing in Phase-2+.
        return torch.zeros_like(h_chart)

    def apply_free_adapter(self, h_chart: Tensor) -> Tensor:
        """
        Apply free-adapter.

        Args:
            h_chart: pre-adapter features [B, D].

        Returns:
            delta_free of shape [B, D].
        """
        if self.free_adapter is None:
            return torch.zeros_like(h_chart)
        return self.free_adapter(h_chart)

    def combine_residuals(
        self,
        delta_task: Optional[Tensor],
        delta_chart: Optional[Tensor],
        delta_free: Optional[Tensor],
        routing_info: Optional[RoutingOutput] = None,
    ) -> Tensor:
        """
        Combine residuals from different adapter sources.

        Combination depends on adapter_mode:
        - TASK_TRAIN: only delta_task.
        - DISTILL: only delta_task (for collection).
        - INFER: delta_chart + gate * delta_free, with fallback.

        Args:
            delta_task: task-adapter residual [B, D].
            delta_chart: chart-adapter residual [B, D].
            delta_free: free-adapter residual [B, D].
            routing_info: routing decisions for gating.

        Returns:
            Combined residual delta_total of shape [B, D].
        """
        if self.adapter_mode == TASK_TRAIN:
            return delta_task if delta_task is not None else torch.zeros(1)
        elif self.adapter_mode == INFER:
            total = torch.zeros(1)
            if delta_chart is not None:
                total = total + delta_chart
            if delta_free is not None and self.use_free_adapter:
                total = total + delta_free
            return total
        else:
            return delta_task if delta_task is not None else torch.zeros(1)

    def set_adapter_mode(self, mode: str) -> None:
        """
        Set the adapter mode.

        Args:
            mode: one of TASK_TRAIN, DISTILL, INFER.
        """
        if mode not in (TASK_TRAIN, DISTILL, INFER):
            raise ValueError(f"Unknown adapter_mode: {mode}")
        self.adapter_mode = mode

    def register_chart_adapter(
        self,
        chart_id: int,
        slot_id: int,
        adapter: nn.Module,
    ) -> None:
        """
        Register a chart-adapter for a specific (chart, slot) pair.

        Args:
            chart_id: chart id.
            slot_id: slot id.
            adapter: ChartAdapter module.
        """
        key = f"{chart_id}_{slot_id}"
        self.chart_adapters[key] = adapter

    def remove_task_adapter(self) -> None:
        """Remove and free the task-adapter."""
        self.task_adapter = None

    def freeze_permanent_adapters(self) -> None:
        """Freeze all permanent adapters (chart, free, routers)."""
        raise NotImplementedError("Phase-0 skeleton only.")

    def unfreeze_task_adapter(self) -> None:
        """Unfreeze task-adapter for training."""
        raise NotImplementedError("Phase-0 skeleton only.")

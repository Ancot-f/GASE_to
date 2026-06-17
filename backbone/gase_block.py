"""
GASEAtlasBlock: ViT transformer block augmented with GASE adapters.

Key modes:
- "task_train": uses task_adapter for teacher residual generation.
  Residual is applied ONLY to the CLS token, not patch tokens.
- "distill": placeholder for feature collection (Phase-3+).
- "infer": uses chart-adapter/free-adapter (Phase-3+); identity in Phase-2.

CRITICAL: h_chart is the pre-adapter CLS feature on the permanent path.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from .gase_components import TASK_TRAIN, DISTILL, INFER, L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT, RoutingOutput


class GASEAtlasBlock(nn.Module):
    """
    A ViT block augmented with GASE atlas capabilities.

    Wraps an original timm ViT block. In task_train mode, applies
    a TaskAdapter residual to the CLS token only. Chart/slot/free
    adapters are Phase-3+.

    Attributes:
        layer_id: ViT block index (e.g., 9, 10, 11).
        dim: feature dimension D.
        original_block: the original timm ViT Block.
        task_adapter: temporary per-task adapter (TaskAdapter).
        chart_router: routes features to charts (Phase-3+).
        slot_router: routes to slots within a chart (Phase-3+).
        chart_adapters: dict (chart_id, slot_id) -> ChartAdapter (Phase-3+).
        free_adapter: FreeAdapter for residual leftover (Phase-3+).
        adapter_mode: current mode.
    """

    def __init__(
        self,
        original_block: nn.Module,
        layer_id: int,
        dim: int,
        config: dict,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.original_block = original_block

        # Adapters (created on demand)
        self.task_adapter: Optional[nn.Module] = None
        self.chart_router: Optional[nn.Module] = None
        self.slot_router: Optional[nn.Module] = None
        self.chart_adapters: Dict[str, nn.Module] = nn.ModuleDict()
        self.chart_states: Dict[int, object] = {}  # chart_id -> ChartState
        self.free_adapter: Optional[nn.Module] = None

        self.adapter_mode: str = INFER

        routing_cfg = config.get("routing", {})
        self.use_free_adapter: bool = config.get("free_adapter", {}).get("enabled", True)
        self.use_soft_chart_routing: bool = routing_cfg.get("use_soft_chart_routing", True)
        self.use_identity_fallback: bool = routing_cfg.get("use_identity_fallback", True)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        return_routing: bool = False,
    ) -> Tuple[Tensor, Optional[RoutingOutput]]:
        """
        Forward pass through the GASE block.

        1. Pass x through the original ViT block.
        2. Extract h_chart (CLS token).
        3. Apply adapter residual based on mode:
           - task_train: task_adapter residual to CLS token.
           - l9_chart_student: L9 uses chart-adapter, L10/L11 use task_adapter.
           - sequential_chart_student: any committed layer uses chart-adapter,
             uncommitted layers use task_adapter.
           - distill/infer: identity pass-through.

        Args:
            x: input features of shape [B, N, D].
            return_routing: if True, also return routing info.

        Returns:
            Tuple of (output [B, N, D], optional RoutingOutput).
        """
        x = self.forward_original_block(x)

        if self.adapter_mode == TASK_TRAIN and self.task_adapter is not None:
            h_chart = self.get_router_feature(x)
            delta_task = self.apply_task_adapter(h_chart)
            x = self.add_delta_to_cls(x, delta_task)

        elif self.adapter_mode in (L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT):
            h_chart = self.get_router_feature(x)
            if self.has_active_chart_adapter():
                delta_chart = self.apply_first_chart_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta_chart)
            elif self.task_adapter is not None:
                delta_task = self.apply_task_adapter(h_chart)
                x = self.add_delta_to_cls(x, delta_task)

        routing_info = None
        return x, routing_info

    def forward_original_block(self, x: Tensor) -> Tensor:
        """
        Forward through the original timm ViT block (attention + MLP).

        Args:
            x: input features [B, N, D].

        Returns:
            Output features [B, N, D].
        """
        return self.original_block(x)

    def add_delta_to_cls(self, x: Tensor, delta: Tensor) -> Tensor:
        """
        Add adapter residual to CLS token only.

        Args:
            x: features of shape [B, N, D] or [B, D].
            delta: residual of shape [B, D].

        Returns:
            Tensor with the same shape as x, with delta added to CLS.
        """
        if x.dim() == 3:
            # [B, N, D] — add to CLS token (position 0) only
            x = x.clone()
            x[:, 0, :] = x[:, 0, :] + delta
            return x
        elif x.dim() == 2:
            # [B, D] — add directly
            return x + delta
        else:
            raise ValueError(f"Expected x of shape [B, N, D] or [B, D], got {x.shape}")

    # ------------------------------------------------------------------
    #  Router feature
    # ------------------------------------------------------------------

    def get_router_feature(self, x: Tensor) -> Tensor:
        """
        Extract the CLS-token feature used for routing.

        h_chart is the pre-current-adapter feature on the permanent path,
        i.e., the output of the ORIGINAL block before any adapter residual.

        Args:
            x: features after original block, shape [B, N, D] or [B, D].

        Returns:
            h_chart of shape [B, D] (CLS token or full feature).
        """
        if x.dim() == 3:
            return x[:, 0]  # CLS token
        elif x.dim() == 2:
            return x
        else:
            raise ValueError(f"Expected x of shape [B, N, D] or [B, D], got {x.shape}")

    # ------------------------------------------------------------------
    #  Adapter application
    # ------------------------------------------------------------------

    def apply_task_adapter(self, h_chart: Tensor) -> Tensor:
        """
        Apply task-adapter in task_train mode.

        Args:
            h_chart: pre-adapter CLS features [B, D].

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
        Apply chart-adapters with routing (Phase-3+).

        Args:
            h_chart: pre-adapter features [B, D].
            chart_probs: chart probabilities [B, num_charts].
            slot_probs: slot probabilities [B, num_slots].

        Returns:
            delta_chart of shape [B, D].
        """
        # Phase-2 placeholder
        raise NotImplementedError("Phase-2 does not implement chart routing.")

    def apply_free_adapter(self, h_chart: Tensor) -> Tensor:
        """
        Apply free-adapter (Phase-3+).

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
        Combine residuals from different adapter sources (Phase-3+).

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
        else:
            total = torch.zeros(1)
            if delta_chart is not None:
                total = total + delta_chart
            if delta_free is not None and self.use_free_adapter:
                total = total + delta_free
            return total

    # ------------------------------------------------------------------
    #  Feature extraction for collection (Phase-3)
    # ------------------------------------------------------------------

    def extract_h_chart_and_delta_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run original block, extract h_chart, and compute task-adapter teacher residual.

        This method is used by FeatureCollector. It does NOT add the residual
        back to the block output — it only computes and returns the values.

        Args:
            x: input features of shape [B, N, D] or [B, D].

        Returns:
            block_output: Tensor with same shape as x after original_block,
                          before adding any adapter residual.
            h_chart: CLS feature of shape [B, D] seen by the router.
            delta_teacher: task_adapter(h_chart) of shape [B, D],
                           or zeros if task_adapter is None.
        """
        block_output = self.forward_original_block(x)
        h_chart = self.get_router_feature(block_output)
        delta_teacher = self.apply_task_adapter(h_chart)
        return block_output, h_chart, delta_teacher

    # ------------------------------------------------------------------
    #  Mode & registration
    # ------------------------------------------------------------------

    def set_adapter_mode(self, mode: str) -> None:
        """
        Set the adapter mode.

        Args:
            mode: one of TASK_TRAIN, DISTILL, INFER,
                  L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT.
        """
        if mode not in (TASK_TRAIN, DISTILL, INFER, L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT):
            raise ValueError(f"Unknown adapter_mode: {mode}")
        self.adapter_mode = mode

    def has_active_chart_adapter(self) -> bool:
        """
        Return True if at least one chart-adapter with chart state is registered.

        Phase-5 assumes one chart and one slot per layer.
        """
        if len(self.chart_adapters) == 0 or len(self.chart_states) == 0:
            return False
        # Check that chart_id=0 has valid mu
        cs = self.chart_states.get(0)
        return cs is not None and getattr(cs, "mu", None) is not None

    def apply_first_chart_adapter(self, h_chart: Tensor) -> Tensor:
        """
        Phase-5: use the first registered chart-adapter with chart mu.

        Args:
            h_chart: pre-adapter CLS features [B, D].

        Returns:
            delta_chart of shape [B, D], or zeros if no active adapter.
        """
        if not self.has_active_chart_adapter():
            return torch.zeros_like(h_chart)
        first_key = next(iter(self.chart_adapters))
        adapter = self.chart_adapters[first_key]
        chart_state = self.chart_states.get(0)
        mu = chart_state.mu.to(h_chart.device)
        return adapter(h_chart, mu)

    def register_chart(self, chart_state: object) -> None:
        """Register a ChartState for this block's atlas."""
        self.chart_states[chart_state.chart_id] = chart_state

    def register_chart_adapter(
        self,
        chart_id: int,
        slot_id: int,
        adapter: nn.Module,
    ) -> None:
        """Register a chart-adapter for a (chart, slot) pair."""
        key = f"{chart_id}_{slot_id}"
        self.chart_adapters[key] = adapter

    def remove_task_adapter(self) -> None:
        """Remove and free the task-adapter."""
        self.task_adapter = None

    def freeze_permanent_adapters(self) -> None:
        """Freeze all permanent adapters (chart, free, routers)."""
        for adapter in self.chart_adapters.values():
            for p in adapter.parameters():
                p.requires_grad = False
        if self.free_adapter is not None:
            for p in self.free_adapter.parameters():
                p.requires_grad = False

    def unfreeze_task_adapter(self) -> None:
        """Unfreeze task-adapter for training."""
        if self.task_adapter is not None:
            for p in self.task_adapter.parameters():
                p.requires_grad = True

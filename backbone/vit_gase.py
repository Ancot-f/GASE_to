"""
ViTGASE: ViT backbone wrapper for GASE-Atlas continual learning.

This class wraps a standard ViT and injects GASEAtlasBlocks at
specified layers (default: L9, L10, L11).

Does NOT modify vit_sema.py — this is an independent GASE backbone.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from .gase_components import TASK_TRAIN, DISTILL, INFER


class ViTGASE(nn.Module):
    """
    ViT wrapper with GASE atlas augmentation.

    Replaces standard ViT blocks at atlas_layers with GASEAtlasBlocks
    that support task-adapter training, chart/slot routing, and
    chart/free adapter inference.

    Attributes:
        backbone_name: name of the base ViT model.
        num_classes: number of output classes.
        embed_dim: embedding dimension D.
        atlas_layers: list of layer indices with GASE blocks.
        blocks: sequential container of all ViT blocks.
        classifier: final classification head.
        use_cls_token: whether to use CLS token for classification.
        adapter_mode: current adapter mode for all GASE blocks.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        num_classes: int = 0,
        embed_dim: int = 768,
        atlas_layers: Optional[List[int]] = None,
        config: Optional[dict] = None,
    ):
        """
        Args:
            backbone_name: base ViT model name.
            num_classes: number of output classes (0 for feature extractor).
            embed_dim: feature dimension D.
            atlas_layers: list of layer indices to augment.
            config: GASE configuration dict.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.atlas_layers = atlas_layers or [9, 10, 11]
        self.config = config or {}

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.classifier: Optional[nn.Module] = None
        self.use_cls_token: bool = True
        self.adapter_mode: str = INFER
        self.out_dim: int = embed_dim

        # build_backbone() is deferred to Phase-2+ when real ViT loading is needed.

    def build_backbone(self) -> None:
        """
        Build the ViT backbone and inject GASE blocks.

        This is a skeleton — the actual implementation will load
        a pretrained ViT and replace specified layers.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def inject_gase_blocks(self) -> None:
        """
        Replace standard blocks at atlas_layers with GASEAtlasBlocks.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the GASE ViT.

        Args:
            x: input images of shape [B, C, H, W].
            return_features: if True, also return intermediate features.

        Returns:
            Dict with keys: "logits" (or "features"), plus optional
            "layer_features", "routing_info".
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Extract features (before classifier head).

        Args:
            x: input images [B, C, H, W].

        Returns:
            Features of shape [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def forward_until_layer(self, x: Tensor, layer_id: int) -> Tensor:
        """
        Forward pass up to (but not including) a specific layer.

        Args:
            x: input images [B, C, H, W].
            layer_id: stop after this layer index.

        Returns:
            Hidden states at layer_id of shape [B, N, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def forward_from_layer(
        self,
        h: Tensor,
        start_layer_id: int,
    ) -> Tensor:
        """
        Forward pass starting from a specific layer.

        Args:
            h: hidden states [B, N, D].
            start_layer_id: first layer to process.

        Returns:
            Output features [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def set_adapter_mode(self, mode: str) -> None:
        """
        Set adapter mode on all GASE blocks.

        Args:
            mode: one of TASK_TRAIN, DISTILL, INFER.
        """
        if mode not in (TASK_TRAIN, DISTILL, INFER):
            raise ValueError(f"Unknown adapter_mode: {mode}")
        self.adapter_mode = mode
        for blk in self.blocks:
            if hasattr(blk, "set_adapter_mode"):
                blk.set_adapter_mode(mode)

    def enable_task_adapters(self) -> None:
        """Enable task-adapter mode on all GASE blocks."""
        self.set_adapter_mode(TASK_TRAIN)

    def disable_task_adapters(self) -> None:
        """Switch to inference mode (disable task-adapters)."""
        self.set_adapter_mode(INFER)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (not adapters)."""
        raise NotImplementedError("Phase-0 skeleton only.")

    def unfreeze_task_adapters(self) -> None:
        """Unfreeze only task-adapter parameters."""
        raise NotImplementedError("Phase-0 skeleton only.")

    def get_atlas_blocks(self) -> List[nn.Module]:
        """
        Return the list of GASEAtlasBlock modules.

        Returns:
            List of GASEAtlasBlock instances.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def get_block(self, layer_id: int) -> nn.Module:
        """
        Get a specific ViT block by layer index.

        Args:
            layer_id: ViT block index.

        Returns:
            The block module at that layer.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

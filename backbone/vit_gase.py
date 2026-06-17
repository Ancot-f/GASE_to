"""
ViTGASE: ViT backbone wrapper for GASE-Atlas continual learning.

Wraps a standard timm ViT and injects GASEAtlasBlocks at specified layers
(default: L9, L10, L11). Supports task_train, distill, and infer modes.

Does NOT modify vit_sema.py — this is an independent GASE backbone.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import timm

from .gase_block import GASEAtlasBlock
from .gase_components import TASK_TRAIN, DISTILL, INFER


class ViTGASE(nn.Module):
    """
    ViT wrapper with GASE atlas augmentation.

    Uses timm to load a pretrained ViT, then replaces blocks at
    atlas_layers with GASEAtlasBlock wrappers. The original ViT block
    is preserved inside each GASEAtlasBlock as original_block.

    Attributes:
        backbone_name: timm model name (e.g. "vit_base_patch16_224").
        embed_dim: feature dimension D (default 768).
        atlas_layers: list of layer indices with GASE blocks.
        blocks: ModuleList of all ViT blocks (some wrapped as GASEAtlasBlock).
        head: final linear classifier.
        adapter_mode: current adapter mode for all GASE blocks.
        out_dim: output feature dimension.
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
            backbone_name: timm ViT model name.
            num_classes: number of output classes.
            embed_dim: feature dimension D.
            atlas_layers: list of layer indices to augment with GASE.
            config: GASE configuration dict.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.atlas_layers = atlas_layers or [9, 10, 11]
        self.config = config or {}
        self.adapter_mode: str = INFER
        self.out_dim: int = embed_dim

        self.build_backbone()

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def build_backbone(self) -> None:
        """
        Build the ViT backbone via timm and inject GASE blocks at atlas_layers.

        Creates a timm ViT without network access (pretrained=False), then
        loads weights from a local safetensors checkpoint if available.
        Replaces blocks at self.atlas_layers with GASEAtlasBlock wrappers.
        """
        # Build ViT without downloading (weights loaded from local file)
        base_vit = timm.create_model(
            self.backbone_name, pretrained=False, num_classes=0
        )

        # Try to load pretrained weights from local checkpoint
        pretrained_path = self.config.get(
            "pretrained_path",
            "/sdd1/syc/My_code/common/pre-model/1k/model.safetensors",
        )
        self._load_pretrained_weights(base_vit, pretrained_path)

        self.patch_embed = base_vit.patch_embed
        self.cls_token = base_vit.cls_token
        self.pos_embed = base_vit.pos_embed
        self.pos_drop = base_vit.pos_drop
        self.norm = base_vit.norm
        self.pre_logits = getattr(base_vit, "pre_logits", nn.Identity())

        # Build block list, injecting GASEAtlasBlock at atlas_layers
        num_blocks = len(base_vit.blocks)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            orig_blk = base_vit.blocks[i]
            if i in self.atlas_layers:
                gase_blk = GASEAtlasBlock(orig_blk, layer_id=i, dim=self.embed_dim, config=self.config)
                self.blocks.append(gase_blk)
            else:
                self.blocks.append(orig_blk)

        # Classifier head
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Free base_vit reference so it can be GC'd (we extracted what we need)
        del base_vit

    def _load_pretrained_weights(self, model: nn.Module, path: str) -> None:
        """
        Load pretrained weights from a local safetensors file.

        The checkpoint keys match timm's internal ViT block format directly
        (fused qkv, mlp.fc1/fc2), so no remapping is needed.

        Args:
            model: timm ViT model with randomly initialized weights.
            path: path to .safetensors checkpoint.
        """
        import os
        from safetensors.torch import load_file as safetensors_load

        if not os.path.exists(path):
            print(f"[ViTGASE] Pretrained weights not found at {path}, using random init.")
            return

        state_dict = safetensors_load(path)

        # Load with strict=False: head, patch_embed, pos_embed may differ
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[ViTGASE] Loaded pretrained weights from {path}")
        if msg.missing_keys:
            print(f"[ViTGASE] Missing keys ({len(msg.missing_keys)}): "
                  f"{msg.missing_keys[:3]}...")
        if msg.unexpected_keys:
            print(f"[ViTGASE] Unexpected keys ({len(msg.unexpected_keys)}): "
                  f"{msg.unexpected_keys[:3]}...")

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Full forward pass: patch embed -> blocks -> norm -> CLS -> head.

        Args:
            x: input images of shape [B, C, H, W].

        Returns:
            Dict with keys "features" (pre-logits CLS token [B, D])
            and "logits" [B, num_classes].
        """
        features = self.forward_features(x)
        logits = self.head(features)
        return {"features": features, "logits": logits}

    def forward_features(self, x: Tensor) -> Tensor:
        """
        Extract CLS-token features through all blocks and final norm.

        Args:
            x: input images [B, C, H, W].

        Returns:
            CLS-token features of shape [B, D].
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock):
                x, _routing = blk(x)
            else:
                x = blk(x)

        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        return x

    def forward_until_layer(self, x: Tensor, layer_id: int) -> Tensor:
        """
        Forward pass through blocks 0..layer_id (inclusive).

        Args:
            x: input images [B, C, H, W].
            layer_id: last layer to process (0-indexed).

        Returns:
            Hidden states after layer_id of shape [B, N, D].
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if isinstance(blk, GASEAtlasBlock):
                x, _routing = blk(x)
            else:
                x = blk(x)
            if i == layer_id:
                break
        return x

    def forward_from_layer(self, h: Tensor, start_layer_id: int) -> Tensor:
        """
        Forward pass from start_layer_id through remaining blocks and norm.

        Args:
            h: hidden states [B, N, D].
            start_layer_id: first layer to process.

        Returns:
            CLS-token features of shape [B, D].
        """
        for i in range(start_layer_id, len(self.blocks)):
            blk = self.blocks[i]
            if isinstance(blk, GASEAtlasBlock):
                h, _routing = blk(h)
            else:
                h = blk(h)
        h = self.norm(h)
        return h[:, 0]

    # ------------------------------------------------------------------
    #  Adapter mode
    # ------------------------------------------------------------------

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
            if isinstance(blk, GASEAtlasBlock):
                blk.set_adapter_mode(mode)

    def enable_task_adapters(self) -> None:
        """Enable task-adapter mode on all GASE blocks."""
        self.set_adapter_mode(TASK_TRAIN)

    def disable_task_adapters(self) -> None:
        """Switch to inference mode (disable task-adapters)."""
        self.set_adapter_mode(INFER)

    # ------------------------------------------------------------------
    #  Parameter management
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all parameters except task adapters and head."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_task_adapters(self) -> None:
        """Unfreeze only task-adapter parameters."""
        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock) and blk.task_adapter is not None:
                for p in blk.task_adapter.parameters():
                    p.requires_grad = True

    def unfreeze_head(self) -> None:
        """Unfreeze classifier head."""
        for p in self.head.parameters():
            p.requires_grad = True

    def freeze_permanent_adapters(self) -> None:
        """Freeze chart adapters, free adapters, routers (not task adapters)."""
        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock):
                for adapter in blk.chart_adapters.values():
                    for p in adapter.parameters():
                        p.requires_grad = False
                if blk.free_adapter is not None:
                    for p in blk.free_adapter.parameters():
                        p.requires_grad = False

    # ------------------------------------------------------------------
    #  Accessors
    # ------------------------------------------------------------------

    def get_atlas_blocks(self) -> List[GASEAtlasBlock]:
        """Return all GASEAtlasBlock instances."""
        return [blk for blk in self.blocks if isinstance(blk, GASEAtlasBlock)]

    def get_block(self, layer_id: int) -> nn.Module:
        """Get a block by layer index."""
        if layer_id < 0 or layer_id >= len(self.blocks):
            raise IndexError(f"layer_id {layer_id} out of range [0, {len(self.blocks)})")
        return self.blocks[layer_id]

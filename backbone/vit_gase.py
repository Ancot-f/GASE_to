"""
ViTGASE: ViT backbone wrapper for GASE-Atlas continual learning.

Phase-6.5: L0-L11 all wrapped as GASEAtlasBlock.
  L0-L8: base_adapter (frozen after Task0), no chart/slot
  L9-L11: task_adapter → chart/slot/chart_adapter/free_adapter
"""

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import timm

from .gase_block import GASEAtlasBlock
from .gase_components import (
    TASK_TRAIN, TASK0_BOOTSTRAP, BASE_PLUS_TASK_TRAIN,
    DISTILL, INFER,
    L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT,
    CURRENT_SLOT_STUDENT, ORACLE_SLOT_STUDENT, KEY_SLOT_STUDENT,
    PATH_KEY_SLOT_STUDENT,
)
from gase.adapters.adapter_factory import build_task_adapter

_ALL_MODES = (
    TASK_TRAIN, TASK0_BOOTSTRAP, BASE_PLUS_TASK_TRAIN,
    DISTILL, INFER,
    L9_CHART_STUDENT, SEQUENTIAL_CHART_STUDENT,
    CURRENT_SLOT_STUDENT, ORACLE_SLOT_STUDENT, KEY_SLOT_STUDENT,
    PATH_KEY_SLOT_STUDENT,
)


class ViTGASE(nn.Module):
    """ViT wrapper with GASE atlas + base adapter augmentation."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        num_classes: int = 0,
        embed_dim: int = 768,
        atlas_layers: Optional[List[int]] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.config = config or {}
        self.atlas_layers = atlas_layers or [9, 10, 11]
        self.base_adapter_layers = self.config.get("base_adapter_layers", [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.bootstrap_adapter_layers = self.config.get("bootstrap_adapter_layers", list(range(12)))
        self.adapter_mode: str = INFER
        self.out_dim: int = embed_dim

        self.build_backbone()

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def build_backbone(self) -> None:
        base_vit = timm.create_model(self.backbone_name, pretrained=False, num_classes=0)
        pretrained_path = self.config.get(
            "pretrained_path", "/sdd1/syc/My_code/common/pre-model/1k/model.safetensors",
        )
        self._load_pretrained_weights(base_vit, pretrained_path)

        self.patch_embed = base_vit.patch_embed
        self.cls_token = base_vit.cls_token
        self.pos_embed = base_vit.pos_embed
        self.pos_drop = base_vit.pos_drop
        self.norm = base_vit.norm
        self.pre_logits = getattr(base_vit, "pre_logits", nn.Identity())

        # L0-L11 all wrapped as GASEAtlasBlock
        num_blocks = len(base_vit.blocks)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            orig_blk = base_vit.blocks[i]
            gase_blk = GASEAtlasBlock(orig_blk, layer_id=i, dim=self.embed_dim, config=self.config)
            self.blocks.append(gase_blk)

        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        del base_vit

    def _load_pretrained_weights(self, model: nn.Module, path: str) -> None:
        import os
        from safetensors.torch import load_file as safetensors_load
        if not os.path.exists(path):
            print(f"[ViTGASE] Pretrained weights not found at {path}, using random init.")
            return
        state_dict = safetensors_load(path)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[ViTGASE] Loaded pretrained weights from {path}")

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.forward_features(x)
        head_out = self.head(features)
        if isinstance(head_out, dict):
            logits = head_out["logits"]
        else:
            logits = head_out
        return {"features": features, "logits": logits}

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x, _routing = blk(x)
            # Phase-7.5: PATH_KEY_SLOT_STUDENT — propagate L9 path to L10/L11
            if (self.adapter_mode == PATH_KEY_SLOT_STUDENT and
                i == self.atlas_layers[0] and isinstance(blk, GASEAtlasBlock)
                and hasattr(blk, "path_slot_id") and blk.path_slot_id is not None):
                for lid in self.atlas_layers:
                    if lid > i:
                        self.blocks[lid].path_slot_id = blk.path_slot_id
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        return x

    def forward_until_layer(self, x: Tensor, layer_id: int) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x, _routing = blk(x)
            if i == layer_id:
                break
        return x

    # ------------------------------------------------------------------
    #  Adapter mode
    # ------------------------------------------------------------------

    def set_adapter_mode(self, mode: str) -> None:
        if mode not in _ALL_MODES:
            raise ValueError(f"Unknown adapter_mode: {mode}")
        self.adapter_mode = mode
        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock):
                blk.set_adapter_mode(mode)

    def enable_task_adapters(self) -> None:
        self.set_adapter_mode(TASK_TRAIN)

    def disable_task_adapters(self) -> None:
        self.set_adapter_mode(INFER)

    # ------------------------------------------------------------------
    #  Base adapter management (Phase-6.5)
    # ------------------------------------------------------------------

    def commit_task0_base_adapters(self) -> None:
        """
        Copy L0-L8 task_adapter → base_adapter, freeze base_adapter,
        remove task_adapter from L0-L8.
        """
        logging.info("[ViTGASE] Committing base adapters for layers %s", self.base_adapter_layers)
        for lid in self.base_adapter_layers:
            blk = self.blocks[lid]
            if blk.task_adapter is not None:
                blk.set_base_adapter(copy.deepcopy(blk.task_adapter), freeze=True)
                blk.task_adapter = None
        logging.info("[ViTGASE] Base adapters committed and frozen.")

    def freeze_base_adapters(self) -> None:
        for lid in self.base_adapter_layers:
            self.blocks[lid].freeze_base_adapter()

    # ------------------------------------------------------------------
    #  Task adapter creation
    # ------------------------------------------------------------------

    def create_task_adapters_for_layers(self, layer_ids: List[int]) -> None:
        dim = self.embed_dim
        for lid in layer_ids:
            blk = self.blocks[lid]
            blk.task_adapter = build_task_adapter(self.config, dim)
        logging.info("[ViTGASE] Created task_adapters for layers %s", layer_ids)

    # ------------------------------------------------------------------
    #  Slot management
    # ------------------------------------------------------------------

    def set_active_slot_id(self, slot_id: Optional[int]) -> None:
        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock):
                blk.set_active_slot_id(slot_id)

    def set_oracle_slot_id(self, slot_id: Optional[int]) -> None:
        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock):
                blk.set_oracle_slot_id(slot_id)

    def set_nll_router(self, router) -> None:
        """Set CalibratedNLLSlotRouter on all atlas blocks."""
        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock):
                blk.nll_router = router

    # ------------------------------------------------------------------
    #  Compute logits helpers
    # ------------------------------------------------------------------

    def compute_teacher_logits(self, images: Tensor) -> Tensor:
        prev_mode = self.adapter_mode
        self.set_adapter_mode(TASK_TRAIN)
        try:
            return self.forward(images)["logits"]
        finally:
            self.set_adapter_mode(prev_mode)

    def compute_oracle_slot_logits(self, images: Tensor, slot_id: int) -> Tensor:
        prev_mode = self.adapter_mode
        self.set_adapter_mode(ORACLE_SLOT_STUDENT)
        self.set_oracle_slot_id(slot_id)
        try:
            return self.forward(images)["logits"]
        finally:
            self.set_adapter_mode(prev_mode)

    def compute_key_slot_logits(self, images: Tensor) -> Tensor:
        self._clear_path_slot_ids()
        prev_mode = self.adapter_mode
        self.set_adapter_mode(KEY_SLOT_STUDENT)
        try:
            return self.forward(images)["logits"]
        finally:
            self.set_adapter_mode(prev_mode)

    def compute_path_key_slot_logits(self, images: Tensor) -> Tensor:
        """Forward with PATH_KEY_SLOT_STUDENT: L9 decides path, L10/L11 follow."""
        self._clear_path_slot_ids()
        prev_mode = self.adapter_mode
        self.set_adapter_mode(PATH_KEY_SLOT_STUDENT)
        try:
            return self.forward(images)["logits"]
        finally:
            self.set_adapter_mode(prev_mode)

    def _clear_path_slot_ids(self) -> None:
        for blk in self.blocks:
            if isinstance(blk, GASEAtlasBlock):
                blk.path_slot_id = None

    def _extract_h_chart_at_layer(self, images: Tensor, layer_id: int) -> Tensor:
        """Extract pre-adapter h_chart (CLS) at a specific layer for routing diagnostics."""
        B = images.shape[0]
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            if i == layer_id:
                blk_out = blk.forward_original_block(x) if isinstance(blk, GASEAtlasBlock) else blk(x)
                if blk_out.dim() == 3:
                    return blk_out[:, 0]
                return blk_out
            x, _routing = blk(x) if isinstance(blk, GASEAtlasBlock) else (blk(x), None)
        return torch.zeros(B, self.embed_dim, device=images.device)

    def collect_last_routing_info(self):
        """Return {per_layer: {lid: info}, path: info_or_none} after forward."""
        per_layer = {}
        path_info = None
        for lid in self.atlas_layers:
            blk = self.blocks[lid]
            if hasattr(blk, "last_routing_info") and blk.last_routing_info is not None:
                per_layer[lid] = blk.last_routing_info
            if (hasattr(blk, "last_path_routing_info") and
                blk.last_path_routing_info is not None):
                path_info = blk.last_path_routing_info
        return {"per_layer": per_layer, "path": path_info}

    def compute_student_logits(self, images: Tensor) -> Tensor:
        prev_mode = self.adapter_mode
        self.set_adapter_mode(SEQUENTIAL_CHART_STUDENT)
        try:
            return self.forward(images)["logits"]
        finally:
            self.set_adapter_mode(prev_mode)

    # ------------------------------------------------------------------
    #  Feature extraction
    # ------------------------------------------------------------------

    def extract_layer_chart_feature_and_teacher(self, images: Tensor, layer_id: int) -> Tuple[Tensor, Tensor]:
        """Extract h_chart + delta_teacher using CURRENT_SLOT_STUDENT for prefix."""
        if layer_id not in self.atlas_layers:
            raise ValueError(f"layer_id {layer_id} not in atlas_layers {self.atlas_layers}")

        prev_mode = self.adapter_mode
        self.set_adapter_mode(CURRENT_SLOT_STUDENT)
        try:
            B = images.shape[0]
            x = self.patch_embed(images)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            for i, blk in enumerate(self.blocks):
                if i == layer_id:
                    _block_output, h_chart, delta_teacher = blk.extract_h_chart_and_delta_teacher(x)
                    return h_chart, delta_teacher
                else:
                    x, _routing = blk(x)
            raise RuntimeError(f"Layer {layer_id} not reached.")
        finally:
            self.set_adapter_mode(prev_mode)

    # ------------------------------------------------------------------
    #  Accessors
    # ------------------------------------------------------------------

    def get_atlas_blocks(self) -> List[GASEAtlasBlock]:
        return [self.blocks[lid] for lid in self.atlas_layers]

    def get_base_blocks(self) -> List[GASEAtlasBlock]:
        return [self.blocks[lid] for lid in self.base_adapter_layers]

    def get_block(self, layer_id: int) -> nn.Module:
        return self.blocks[layer_id]

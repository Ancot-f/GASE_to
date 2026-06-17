"""FeatureCollector: collects per-layer features for distillation.

CRITICAL DESIGN CONSTRAINT:
  h_chart is the pre-current-adapter feature as seen by the router
  on the PERMANENT path (not task-adapter path, not backbone-fixed path).

  For L9: h_chart = CLS token after L9 original_block.
  L10/L11 will require sequential chart-adapter commit first (Phase-4+).

Phase-3: Authoritative L9 collection only.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader


@dataclass
class LayerFeatureBatch:
    """
    Stores collected features for a single layer.

    Attributes:
        layer_id: ViT block index.
        h_chart: pre-adapter features of shape [N, D] (permanent path).
        delta_teacher: teacher residual of shape [N, D].
        teacher_logits: teacher logits of shape [N, C].
        labels: ground-truth labels of shape [N].
        sample_indices: global sample indices of shape [N] (optional).
    """

    layer_id: int
    h_chart: Tensor
    delta_teacher: Optional[Tensor] = None
    teacher_logits: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    sample_indices: Optional[Tensor] = None

    def to(self, device) -> "LayerFeatureBatch":
        """Move all tensor fields to the target device. Returns self."""
        self.h_chart = self.h_chart.to(device)
        if self.delta_teacher is not None:
            self.delta_teacher = self.delta_teacher.to(device)
        if self.teacher_logits is not None:
            self.teacher_logits = self.teacher_logits.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.sample_indices is not None:
            self.sample_indices = self.sample_indices.to(device)
        return self

    def summary(self) -> Dict[str, Any]:
        """
        Return shapes and norm statistics for debugging.

        Returns:
            Dict with keys: layer_id, num_samples, shapes, norm stats, label range.
        """
        info: Dict[str, Any] = {
            "layer_id": self.layer_id,
            "num_samples": self.h_chart.shape[0],
            "h_chart_shape": list(self.h_chart.shape),
            "delta_teacher_shape": (
                list(self.delta_teacher.shape) if self.delta_teacher is not None else None
            ),
            "teacher_logits_shape": (
                list(self.teacher_logits.shape) if self.teacher_logits is not None else None
            ),
            "h_chart_norm_mean": float(self.h_chart.norm(dim=-1).mean().cpu()),
            "delta_teacher_norm_mean": (
                float(self.delta_teacher.norm(dim=-1).mean().cpu())
                if self.delta_teacher is not None else None
            ),
            "label_min": int(self.labels.min().cpu()) if self.labels is not None else None,
            "label_max": int(self.labels.max().cpu()) if self.labels is not None else None,
            "has_sample_indices": self.sample_indices is not None,
        }
        return info


class FeatureCollector:
    """
    Collects per-layer features from the model for chart building,
    slot construction, and adapter distillation.

    Phase-3: Only L9 authoritative collection is implemented.
    L10/L11 require sequential chart-adapter commit (Phase-4+).

    Supports two collect modes:
    - "direct": uses model (GASEVitNet / ViTGASE) directly.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        atlas_layers: List[int],
        device: torch.device,
        collect_mode: str = "sequential",
    ):
        """
        Args:
            model: the GASEVitNet or ViTGASE model.
            atlas_layers: list of layer indices to collect (e.g., [9]).
            device: target device.
            collect_mode: "sequential" (default, Phase-3 only L9).
        """
        self.model = model
        self.atlas_layers = atlas_layers
        self.device = device
        self.collect_mode = collect_mode

    # ------------------------------------------------------------------
    #  L9 collection (Phase-3)
    # ------------------------------------------------------------------

    def collect_l9_features(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> LayerFeatureBatch:
        """
        Collect authoritative L9 features for chart building.

        1. Forward images through L0-L8 blocks.
        2. At L9, run original_block, extract h_chart (CLS token) and delta_teacher.
        3. Compute teacher_logits via full task_train forward.
        4. Store labels and optional sample indices.

        This runs under torch.no_grad() and model.eval().
        Collected tensors are detached and moved to CPU to save GPU memory.

        Args:
            data_loader: DataLoader yielding (idx, images, labels) or (images, labels).
            task_id: current task id (for logging).

        Returns:
            LayerFeatureBatch with all collected data concatenated.
        """
        self.model.eval()

        all_h_chart: List[Tensor] = []
        all_delta_teacher: List[Tensor] = []
        all_teacher_logits: List[Tensor] = []
        all_labels: List[Tensor] = []
        all_sample_indices: List[Tensor] = []

        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch — supports both (idx, images, labels) and (images, labels)
                if len(batch) == 3:
                    sample_idx, inputs, targets = batch
                    if isinstance(sample_idx, Tensor):
                        all_sample_indices.append(sample_idx)
                elif len(batch) == 2:
                    inputs, targets = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Determine the backbone (handle DataParallel wrapper)
                backbone = self.model
                if hasattr(self.model, "module"):
                    backbone = self.model.module
                if hasattr(backbone, "backbone"):
                    backbone = backbone.backbone

                # Extract h_chart and delta_teacher at L9
                h_chart, delta_teacher = backbone.extract_layer_chart_feature_and_teacher(
                    inputs, layer_id=9
                )

                # Compute full teacher logits
                teacher_logits = backbone.compute_teacher_logits(inputs)

                # Detach and move to CPU
                all_h_chart.append(h_chart.detach().cpu())
                all_delta_teacher.append(delta_teacher.detach().cpu())
                all_teacher_logits.append(teacher_logits.detach().cpu())
                all_labels.append(targets.cpu())

        # Concatenate
        h_chart_cat = torch.cat(all_h_chart, dim=0)
        delta_teacher_cat = torch.cat(all_delta_teacher, dim=0)
        teacher_logits_cat = torch.cat(all_teacher_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        sample_indices_cat = (
            torch.cat(all_sample_indices, dim=0) if all_sample_indices else None
        )

        batch = LayerFeatureBatch(
            layer_id=9,
            h_chart=h_chart_cat,
            delta_teacher=delta_teacher_cat,
            teacher_logits=teacher_logits_cat,
            labels=labels_cat,
            sample_indices=sample_indices_cat,
        )

        logging.info(
            "[FeatureCollector] task=%d layer=%d samples=%d",
            task_id, 9, h_chart_cat.shape[0],
        )
        logging.info("[FeatureCollector] %s", batch.summary())

        return batch

    # ------------------------------------------------------------------
    #  General collection entry (Phase-3: delegates to L9)
    # ------------------------------------------------------------------

    def collect_for_task(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> Dict[int, LayerFeatureBatch]:
        """
        Collect features for all atlas layers for a given task.

        Phase-3: only L9 is collected authoritatively.
        L10/L11 raise NotImplementedError.

        Args:
            data_loader: DataLoader.
            task_id: current task id.

        Returns:
            Dict mapping layer_id -> LayerFeatureBatch.
        """
        result: Dict[int, LayerFeatureBatch] = {}
        for lid in self.atlas_layers:
            if lid == 9:
                result[lid] = self.collect_l9_features(data_loader, task_id)
            else:
                raise NotImplementedError(
                    f"Phase-3 only supports L9 collection. "
                    f"Layer {lid} requires sequential chart-adapter commit."
                )
        return result

    def collect_layer_features(
        self,
        data_loader: DataLoader,
        layer_id: int,
        task_id: int,
    ) -> LayerFeatureBatch:
        """
        Collect features at a single layer.

        Phase-3: delegates to collect_l9_features for layer_id=9.

        Args:
            data_loader: DataLoader.
            layer_id: ViT block index.
            task_id: current task id.

        Returns:
            LayerFeatureBatch.
        """
        if layer_id == 9:
            return self.collect_l9_features(data_loader, task_id)
        else:
            raise NotImplementedError(
                f"Phase-3 only supports L9 collection (got layer_id={layer_id})."
            )

    # ------------------------------------------------------------------
    #  Future: sequential collection (Phase-4+)
    # ------------------------------------------------------------------

    def collect_sequential_layer_features(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> Dict[int, List[LayerFeatureBatch]]:
        """Phase-4+: sequential L9→L10→L11 collection."""
        raise NotImplementedError("Phase-4+ will implement sequential collection.")

    def collect_l10_features_after_committed_l9(
        self, data_loader: DataLoader, task_id: int
    ) -> List[LayerFeatureBatch]:
        """Phase-4+: L10 collection after L9 chart-adapter commit."""
        raise NotImplementedError("Phase-4+ will implement L10 collection.")

    def collect_l11_features_after_committed_l9_l10(
        self, data_loader: DataLoader, task_id: int
    ) -> List[LayerFeatureBatch]:
        """Phase-4+: L11 collection after L9+L10 chart-adapter commit."""
        raise NotImplementedError("Phase-4+ will implement L11 collection.")

    def get_h_chart_from_model(self, images: Tensor, layer_id: int) -> Tensor:
        raise NotImplementedError("Phase-3 uses extract_layer_chart_feature_and_teacher.")

    def get_delta_teacher(self, h_chart: Tensor, layer_id: int) -> Tensor:
        raise NotImplementedError("Phase-3 uses extract_layer_chart_feature_and_teacher.")

    def get_teacher_logits(self, images: Tensor) -> Tensor:
        raise NotImplementedError("Phase-3 uses compute_teacher_logits on the backbone.")

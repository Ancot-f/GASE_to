"""FeatureCollector: collects per-layer features for distillation.

CRITICAL DESIGN CONSTRAINT:
  h_chart is the pre-current-adapter feature as seen by the router
  on the PERMANENT path (not task-adapter path, not backbone-fixed path).

  For layers L9/L10/L11, features must be collected SEQUENTIALLY:
    1. Forward to L9, collect h_chart at L9.
    2. Commit L9 chart-adapter, re-forward to L10, collect h_chart at L10.
    3. Commit L9+L10, re-forward to L11, collect h_chart at L11.

  This ensures that each chart layer sees the correct cumulative effect
  of all previous chart-adapters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader


@dataclass
class LayerFeatureBatch:
    """
    Stores one batch of collected features for a single layer.

    Attributes:
        layer_id: ViT block index.
        h_chart: pre-adapter features of shape [B, D] (permanent path).
        delta_teacher: teacher residual of shape [B, D] (optional).
        teacher_logits: teacher logits of shape [B, C] (optional).
        labels: ground-truth labels of shape [B] (optional).
        sample_indices: global sample indices [B] (optional).
    """

    layer_id: int
    h_chart: Tensor
    delta_teacher: Optional[Tensor] = None
    teacher_logits: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    sample_indices: Optional[Tensor] = None


class FeatureCollector:
    """
    Collects per-layer features from the model for chart building,
    slot construction, and adapter distillation.

    Supports two modes:
    - "independent": collect each layer independently (for initial charts).
    - "sequential": collect L9 first, commit, then L10, etc. (for accurate h_chart).
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
            model: the ViTGASE model.
            atlas_layers: list of layer indices to collect (e.g., [9, 10, 11]).
            device: target device.
            collect_mode: "independent" or "sequential".
        """
        self.model = model
        self.atlas_layers = atlas_layers
        self.device = device
        self.collect_mode = collect_mode

    def collect_for_task(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> Dict[int, List[LayerFeatureBatch]]:
        """
        Collect features for all atlas layers for a given task.

        Args:
            data_loader: DataLoader yielding (_, images, labels).
            task_id: current task id.

        Returns:
            Dict mapping layer_id -> list of LayerFeatureBatch.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def collect_layer_features(
        self,
        data_loader: DataLoader,
        layer_id: int,
        task_id: int,
    ) -> List[LayerFeatureBatch]:
        """
        Collect features at a single layer.

        Args:
            data_loader: DataLoader.
            layer_id: ViT block index.
            task_id: current task id.

        Returns:
            List of LayerFeatureBatch per mini-batch.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def collect_sequential_layer_features(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> Dict[int, List[LayerFeatureBatch]]:
        """
        Collect features sequentially across layers.

        For L9: forward to L9, collect h_chart, teacher residual.
        For L10: commit L9 adapters, forward to L10, collect.
        For L11: commit L9+L10 adapters, forward to L11, collect.

        Args:
            data_loader: DataLoader.
            task_id: current task id.

        Returns:
            Dict mapping layer_id -> list of LayerFeatureBatch.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def collect_l9_features(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> List[LayerFeatureBatch]:
        """
        Collect features at L9 (no prior chart-adapter active).

        Args:
            data_loader: DataLoader.
            task_id: current task id.

        Returns:
            List of LayerFeatureBatch for L9.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def collect_l10_features_after_committed_l9(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> List[LayerFeatureBatch]:
        """
        Collect features at L10 after L9 chart-adapters are committed.

        This requires model.forward_until_layer(x, 9) to get h at L9,
        then apply_chart_adapters at L9, then forward to L10.

        Args:
            data_loader: DataLoader.
            task_id: current task id.

        Returns:
            List of LayerFeatureBatch for L10.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def collect_l11_features_after_committed_l9_l10(
        self,
        data_loader: DataLoader,
        task_id: int,
    ) -> List[LayerFeatureBatch]:
        """
        Collect features at L11 after L9 and L10 chart-adapters are committed.

        Args:
            data_loader: DataLoader.
            task_id: current task id.

        Returns:
            List of LayerFeatureBatch for L11.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def get_h_chart_from_model(
        self,
        images: Tensor,
        layer_id: int,
    ) -> Tensor:
        """
        Extract h_chart from the model at a specific layer.

        h_chart is the pre-adapter feature on the permanent inference path,
        NOT the task-adapter path.

        Args:
            images: batch of images [B, C, H, W].
            layer_id: ViT block index.

        Returns:
            h_chart of shape [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def get_delta_teacher(
        self,
        h_chart: Tensor,
        layer_id: int,
    ) -> Tensor:
        """
        Get teacher residual at a specific layer.

        delta_teacher = output_with_task_adapter - output_without_adapter

        Args:
            h_chart: pre-adapter features [B, D].
            layer_id: ViT block index.

        Returns:
            delta_teacher of shape [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def get_teacher_logits(self, images: Tensor) -> Tensor:
        """
        Get teacher logits from the full task-adapter model.

        Args:
            images: batch of images [B, C, H, W].

        Returns:
            Teacher logits of shape [B, C].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

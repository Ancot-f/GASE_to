"""DistillCache: stores and manages collected features for distillation."""

from typing import Dict, List, Optional

from torch import Tensor

from .feature_collector import LayerFeatureBatch


class DistillCache:
    """
    In-memory cache for distillation features.

    Stores per-layer feature batches collected during the
    feature collection phase. Provides efficient access patterns
    for chart building, slot construction, and adapter distillation.

    The cache accumulates data across multiple data-loader passes
    and provides device management for GPU training.
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[int, List[LayerFeatureBatch]] = {}

    def add_layer_batch(self, layer_id: int, batch: LayerFeatureBatch) -> None:
        """
        Add a LayerFeatureBatch to the cache for a given layer.

        Args:
            layer_id: ViT block index.
            batch: LayerFeatureBatch to store.
        """
        if layer_id not in self._cache:
            self._cache[layer_id] = []
        self._cache[layer_id].append(batch)

    def get_layer_cache(self, layer_id: int) -> List[LayerFeatureBatch]:
        """
        Retrieve all cached batches for a layer.

        Args:
            layer_id: ViT block index.

        Returns:
            List of LayerFeatureBatch (empty list if layer not cached).
        """
        return self._cache.get(layer_id, [])

    def clear(self) -> None:
        """Remove all cached data and free memory."""
        self._cache.clear()

    def to_device(self, device) -> None:
        """
        Move all cached tensors to the specified device.

        Args:
            device: torch device.
        """
        for layer_batches in self._cache.values():
            for batch in layer_batches:
                batch.h_chart = batch.h_chart.to(device)
                if batch.delta_teacher is not None:
                    batch.delta_teacher = batch.delta_teacher.to(device)
                if batch.teacher_logits is not None:
                    batch.teacher_logits = batch.teacher_logits.to(device)
                if batch.labels is not None:
                    batch.labels = batch.labels.to(device)

    def summary(self) -> Dict[int, Dict[str, int]]:
        """
        Return a summary of cached data.

        Returns:
            Dict mapping layer_id -> {"num_batches": int, "num_samples": int}.
        """
        summary_dict = {}
        for layer_id, batches in self._cache.items():
            num_samples = sum(b.h_chart.shape[0] for b in batches)
            summary_dict[layer_id] = {
                "num_batches": len(batches),
                "num_samples": num_samples,
            }
        return summary_dict

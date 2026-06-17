"""DistillCache: stores and manages collected features for distillation.

Phase-3: stores one LayerFeatureBatch per layer (single concatenated batch).
Phase-4+: will support multi-pass accumulation via list of batches.
"""

from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from .feature_collector import LayerFeatureBatch


class DistillCache:
    """
    In-memory cache for distillation features.

    Stores per-layer feature batches collected during the
    feature collection phase. Provides efficient access patterns
    for chart building, slot construction, and adapter distillation.

    Phase-3: stores a single LayerFeatureBatch per layer (all samples
    concatenated into one batch). Phase-4+ may add multi-pass support.
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[int, Union[LayerFeatureBatch, List[LayerFeatureBatch]]] = {}

    def add_layer_batch(self, layer_id: int, batch: LayerFeatureBatch) -> None:
        """
        Add a LayerFeatureBatch to the cache for a given layer.

        If a batch already exists for this layer, it is replaced.
        For multi-pass accumulation, use add_layer_batch_list.

        Args:
            layer_id: ViT block index.
            batch: LayerFeatureBatch to store.
        """
        self._cache[layer_id] = batch

    def add_layer_batch_list(self, layer_id: int, batch: LayerFeatureBatch) -> None:
        """
        Append a batch to the list for this layer (multi-pass accumulation).

        Args:
            layer_id: ViT block index.
            batch: LayerFeatureBatch to append.
        """
        if layer_id not in self._cache:
            self._cache[layer_id] = []
        lst = self._cache[layer_id]
        if not isinstance(lst, list):
            lst = [lst]
            self._cache[layer_id] = lst
        lst.append(batch)

    def get_layer_cache(
        self, layer_id: int
    ) -> Optional[Union[LayerFeatureBatch, List[LayerFeatureBatch]]]:
        """
        Retrieve cached data for a layer.

        Args:
            layer_id: ViT block index.

        Returns:
            LayerFeatureBatch, list of batches, or None.
        """
        return self._cache.get(layer_id, None)

    def clear(self) -> None:
        """Remove all cached data and free memory."""
        self._cache.clear()

    def to_device(self, device) -> None:
        """
        Move all cached tensors to the specified device.

        Args:
            device: torch device.
        """
        for value in self._cache.values():
            if isinstance(value, list):
                for batch in value:
                    batch.to(device)
            else:
                value.to(device)

    def summary(self) -> Dict[int, Dict[str, Any]]:
        """
        Return a summary of cached data.

        Returns:
            Dict mapping layer_id -> summary dict with shapes and stats.
        """
        summary_dict: Dict[int, Dict[str, Any]] = {}
        for layer_id, value in self._cache.items():
            if isinstance(value, list):
                num_batches = len(value)
                num_samples = sum(
                    b.h_chart.shape[0] for b in value if b.h_chart is not None
                )
                summary_dict[layer_id] = {
                    "num_batches": num_batches,
                    "num_samples": num_samples,
                    "first_batch_summary": value[0].summary() if value else {},
                }
            else:
                summary_dict[layer_id] = value.summary()
        return summary_dict

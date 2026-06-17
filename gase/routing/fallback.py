"""Fallback mechanisms for when chart/slot routing is uncertain."""

from typing import Optional

from torch import Tensor


class IdentityFallback:
    """
    Identity fallback for uncertain routing.

    When chart/slot routing produces high uncertainty,
    fall back to identity (no residual modification).
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: whether identity fallback is active.
        """
        self.enabled = enabled

    def should_fallback(
        self,
        entropy: Tensor,
        margin: Tensor,
        entropy_threshold: float = 1.0,
        margin_threshold: float = 0.05,
    ) -> Tensor:
        """
        Determine which samples need identity fallback.

        Args:
            entropy: chart assignment entropy [B].
            margin: top-1 vs top-2 margin [B].
            entropy_threshold: max allowed entropy.
            margin_threshold: min required margin.

        Returns:
            Boolean mask of shape [B], True for fallback samples.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def apply(
        self,
        delta: Tensor,
        fallback_mask: Tensor,
    ) -> Tensor:
        """
        Apply identity fallback (zero out residual for masked samples).

        Args:
            delta: residual [B, D].
            fallback_mask: boolean mask [B].

        Returns:
            Modified residual [B, D] with fallback entries zeroed.
        """
        raise NotImplementedError("Phase-0 skeleton only.")


class FreeAdapterFallback:
    """
    Free-adapter fallback for uncertain routing.

    Instead of identity, uses the free-adapter to handle
    samples that don't fit any chart/slot well.
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: whether free-adapter fallback is active.
        """
        self.enabled = enabled

    def should_use_free_adapter(
        self,
        entropy: Tensor,
        margin: Tensor,
        entropy_threshold: float = 1.0,
        margin_threshold: float = 0.05,
    ) -> Tensor:
        """
        Determine which samples should use free-adapter.

        Args:
            entropy: chart assignment entropy [B].
            margin: top-1 vs top-2 margin [B].
            entropy_threshold: max allowed entropy.
            margin_threshold: min required margin.

        Returns:
            Boolean mask of shape [B].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def combine_with_free_adapter(
        self,
        delta_chart: Tensor,
        delta_free: Tensor,
        free_gate: Tensor,
    ) -> Tensor:
        """
        Combine chart-adapter and free-adapter residuals.

        delta = (1 - gate) * delta_chart + gate * delta_free

        Args:
            delta_chart: chart-adapter residual [B, D].
            delta_free: free-adapter residual [B, D].
            free_gate: gate values in [0, 1] of shape [B].

        Returns:
            Combined residual [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

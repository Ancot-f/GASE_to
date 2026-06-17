"""SlotMergeManager: merges redundant slots within a chart."""

from typing import List, Optional, Tuple

from torch import Tensor

from .slot_state import SlotState


class SlotMergeManager:
    """
    Manages slot merge operations.

    Merging combines two slots that produce similar residual
    transformations, reducing redundancy and freeing capacity
    for new slots.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: merge configuration.
        """
        self.config = config

    def find_merge_candidates(
        self,
        slot_states: List[SlotState],
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of slots that are candidates for merging.

        Criteria: similar residual outputs, similar projection bases,
        similar router keys.

        Args:
            slot_states: all slots in a chart.

        Returns:
            List of (slot_id_a, slot_id_b, similarity_score) tuples.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def should_merge(
        self,
        slot_a: SlotState,
        slot_b: SlotState,
        h_anchor: Tensor,
    ) -> bool:
        """
        Decide whether two slots should be merged.

        Args:
            slot_a: first slot.
            slot_b: second slot.
            h_anchor: anchor features [N, D] for evaluating merge quality.

        Returns:
            True if slots should be merged.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def merge_slots(
        self,
        slot_a: SlotState,
        slot_b: SlotState,
        h_anchor: Tensor,
        delta_teacher: Tensor,
    ) -> SlotState:
        """
        Merge two slots into a single slot.

        Averages projection bases on the Grassmann manifold,
        averages linear maps, and assigns a new slot_id.

        Args:
            slot_a: first slot to merge.
            slot_b: second slot to merge.
            h_anchor: anchor features [N, D].
            delta_teacher: teacher residuals [N, D] for refitting.

        Returns:
            New merged SlotState. Original slots should be retired.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

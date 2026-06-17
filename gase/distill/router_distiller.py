"""SlotRouterDistiller: trains slot router to mimic teacher-guided assignment."""

from typing import Dict, List

from torch import Tensor

from ..atlas.chart_state import ChartState
from ..slots.slot_state import SlotState


class SlotRouterDistiller:
    """
    Trains the slot router to predict which slot best explains
    the teacher residual for a given feature.

    During distillation, teacher soft labels are constructed by:
    1. Computing the fit error of each slot for each sample.
    2. Converting fit errors to soft assignment probabilities.
    3. Training the router to match these soft labels via KL divergence.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: distillation config dict.
        """
        self.config = config

    def build_teacher_slot_labels(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        chart_state: ChartState,
        slot_states: List[SlotState],
    ) -> Tensor:
        """
        Build teacher-assigned soft slot labels.

        For each sample, compute fit error against each slot,
        then convert to softmax probabilities.

        Args:
            h_chart: features [B, D].
            delta_teacher: teacher residuals [B, D].
            chart_state: parent chart.
            slot_states: available slots.

        Returns:
            Soft slot labels of shape [B, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_teacher_guided_slot_targets(
        self,
        h_chart: Tensor,
        delta_teacher: Tensor,
        chart_state: ChartState,
        slot_states: List[SlotState],
    ) -> Tensor:
        """
        Compute teacher-guided slot targets using fit error.

        target_scores[s] = -||delta_teacher - A_{c,s}(h)||^2
        targets = softmax(target_scores / temperature)

        Args:
            h_chart: features [B, D].
            delta_teacher: teacher residuals [B, D].
            chart_state: parent chart.
            slot_states: available slots.

        Returns:
            Teacher-guided slot targets of shape [B, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def train_slot_router(
        self,
        router,
        h_chart: Tensor,
        slot_targets: Tensor,
    ) -> None:
        """
        Train slot router via gradient descent.

        Loss = KL(slot_targets || router_probs)

        Args:
            router: TeacherGuidedSlotRouter module.
            h_chart: features [N, D].
            slot_targets: teacher-assigned slot targets [N, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def evaluate_router_stability(
        self,
        router,
        h_chart: Tensor,
        slot_targets: Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate router stability metrics.

        Args:
            router: trained router.
            h_chart: features [N, D].
            slot_targets: teacher targets [N, num_slots].

        Returns:
            Dict with keys: top1_accuracy, top2_accuracy, mean_kl, entropy.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

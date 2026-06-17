"""Slot quality and diagnostic metrics."""

from typing import Dict

from torch import Tensor

from .slot_state import SlotState


def compute_residual_fit_r2(
    h_chart: Tensor,
    delta_teacher: Tensor,
    delta_slot: Tensor,
) -> float:
    """
    Compute R^2 score of slot residual fit.

    R^2 = 1 - ||delta_teacher - delta_slot||^2 / ||delta_teacher||^2

    Args:
        h_chart: features [N, D].
        delta_teacher: teacher residuals [N, D].
        delta_slot: slot residuals [N, D].

    Returns:
        R^2 score (higher is better, max 1.0).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_centered_sub_r2(
    h_chart: Tensor,
    delta_teacher: Tensor,
    delta_slot: Tensor,
    chart_mean: Tensor,
) -> float:
    """
    Compute R^2 after centering by chart mean.

    Args:
        h_chart: features [N, D].
        delta_teacher: teacher residuals [N, D].
        delta_slot: slot residuals [N, D].
        chart_mean: chart mean [D].

    Returns:
        Centered R^2 score.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_residual_cosine(
    delta_teacher: Tensor,
    delta_slot: Tensor,
) -> Tensor:
    """
    Compute cosine similarity between teacher and slot residuals.

    Args:
        delta_teacher: teacher residuals [B, D].
        delta_slot: slot residuals [B, D].

    Returns:
        Cosine similarity per sample of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_logit_kl_score(
    logits_teacher: Tensor,
    logits_student: Tensor,
    temperature: float = 1.0,
) -> float:
    """
    Compute KL divergence between teacher and student logits.

    Args:
        logits_teacher: teacher logits [B, C].
        logits_student: student logits [B, C].
        temperature: softening temperature.

    Returns:
        Scalar KL divergence.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_slot_quality(
    slot_state: SlotState,
    h_chart: Tensor,
    delta_teacher: Tensor,
    delta_chart: Tensor,
) -> Dict[str, float]:
    """
    Compute comprehensive slot quality metrics.

    Args:
        slot_state: slot to evaluate.
        h_chart: features assigned to slot [N, D].
        delta_teacher: teacher residuals [N, D].
        delta_chart: chart-adapter residuals [N, D].

    Returns:
        Dict with keys: residual_fit_r2, residual_cosine_mean,
        usage_rate, support, etc.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_slot_usage_entropy(
    slot_probs: Tensor,
) -> float:
    """
    Compute entropy of slot usage distribution across samples.

    Higher entropy = more uniform slot usage (better diversity).

    Args:
        slot_probs: slot probabilities [B, num_slots].

    Returns:
        Scalar entropy value.
    """
    raise NotImplementedError("Phase-0 skeleton only.")

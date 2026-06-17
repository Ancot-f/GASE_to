"""Lightweight components shared by GASE blocks and adapters."""

from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor

# Adapter mode constants
TASK_TRAIN = "task_train"
DISTILL = "distill"
INFER = "infer"

# Chart state constants
CHART_CANDIDATE = "candidate"
CHART_PROVISIONAL = "provisional"
CHART_ACTIVE = "active"
CHART_MATURE = "mature"
CHART_SATURATED = "saturated"
CHART_DORMANT = "dormant"

# Slot state constants
SLOT_CANDIDATE = "candidate"
SLOT_ACTIVE = "active"
SLOT_MATURE = "mature"
SLOT_MERGED = "merged"
SLOT_RETIRED = "retired"


@dataclass
class ResidualOutput:
    """
    Residual outputs from different adapter types for one GASE block.

    Attributes:
        delta_task: task-adapter residual of shape [B, D].
        delta_chart: chart-adapter residual of shape [B, D].
        delta_free: free-adapter residual of shape [B, D].
        delta_total: combined residual of shape [B, D].
    """

    delta_task: Optional[Tensor] = None
    delta_chart: Optional[Tensor] = None
    delta_free: Optional[Tensor] = None
    delta_total: Optional[Tensor] = None


@dataclass
class RoutingOutput:
    """
    Routing decisions for one GASE block.

    Attributes:
        chart_probs: p(chart | h) of shape [B, num_charts].
        slot_probs: p(slot | h, chart) of shape [B, num_slots].
        selected_chart_ids: top-m chart ids of shape [B, top_m].
        selected_slot_ids: top-k slot ids of shape [B, top_k].
        free_gate: free-adapter gate values of shape [B] in [0, 1].
        fallback_mask: boolean mask of shape [B] for identity fallback.
    """

    chart_probs: Optional[Tensor] = None
    slot_probs: Optional[Tensor] = None
    selected_chart_ids: Optional[Tensor] = None
    selected_slot_ids: Optional[Tensor] = None
    free_gate: Optional[Tensor] = None
    fallback_mask: Optional[Tensor] = None

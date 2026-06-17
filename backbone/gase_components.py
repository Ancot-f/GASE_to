"""Lightweight components shared by GASE blocks and adapters."""

from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor

# Adapter mode constants
TASK_TRAIN = "task_train"
TASK0_BOOTSTRAP = "task0_bootstrap"
BASE_PLUS_TASK_TRAIN = "base_plus_task_train"
DISTILL = "distill"
INFER = "infer"
L9_CHART_STUDENT = "l9_chart_student"
SEQUENTIAL_CHART_STUDENT = "sequential_chart_student"
CURRENT_SLOT_STUDENT = "current_slot_student"
ORACLE_SLOT_STUDENT = "oracle_slot_student"
KEY_SLOT_STUDENT = "key_slot_student"
PATH_KEY_SLOT_STUDENT = "path_key_slot_student"

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
    delta_task: Optional[Tensor] = None
    delta_chart: Optional[Tensor] = None
    delta_free: Optional[Tensor] = None
    delta_total: Optional[Tensor] = None


@dataclass
class RoutingOutput:
    chart_probs: Optional[Tensor] = None
    slot_probs: Optional[Tensor] = None
    selected_chart_ids: Optional[Tensor] = None
    selected_slot_ids: Optional[Tensor] = None
    free_gate: Optional[Tensor] = None
    fallback_mask: Optional[Tensor] = None

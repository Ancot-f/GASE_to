"""SlotState: dataclass for storing per-slot parameters and metadata."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class SlotState:
    """
    State of a single slot within a chart.

    A slot represents a reusable residual transformation mode.
    It stores projection bases (P, R), a low-rank linear map (B),
    a bias (b), and a router key vector for key-based slot routing.

    Slots are NOT task identities; a single task may use multiple slots,
    and a single slot may serve multiple tasks.

    Attributes:
        slot_id: unique slot identifier within a chart.
        chart_id: chart this slot belongs to.
        layer_id: ViT block index.
        input_rank: rank of input projection P.
        output_rank: rank of output projection R.
        P: input projection basis of shape [D, input_rank].
        R: output projection basis of shape [output_rank, D].
        B: low-rank linear map of shape [output_rank, input_rank].
        b: residual bias of shape [D].
        key: router key vector of shape [D] for key-based routing.
        support: number of samples assigned to this slot.
        quality: dict of quality metrics.
        state: lifecycle state (candidate/active/mature/merged/retired).
        created_task_id: task that created this slot.
        last_updated_task_id: last task that updated this slot.
        used_count: number of times this slot was selected during routing.
    """

    slot_id: int
    chart_id: int
    layer_id: int
    input_rank: int = 8
    output_rank: int = 4
    P: Optional[Tensor] = None
    R: Optional[Tensor] = None
    B: Optional[Tensor] = None
    b: Optional[Tensor] = None
    key: Optional[Tensor] = None
    key_var: Optional[Tensor] = None
    router_key: Optional[Tensor] = None      # [r_q] shared Q-space key
    router_var: Optional[Tensor] = None      # [r_q] shared Q-space variance
    router_support: int = 0
    support: int = 0
    quality: Dict[str, float] = field(default_factory=dict)
    state: str = "candidate"
    created_task_id: int = 0
    last_updated_task_id: int = 0
    used_count: int = 0

    def is_active(self) -> bool:
        """Return True if slot is in active or mature state."""
        return self.state in ("active", "mature")

    def is_mature(self) -> bool:
        """Return True if slot has reached mature state."""
        return self.state == "mature"

    def can_update(self) -> bool:
        """Return True if slot can receive further updates."""
        return self.state in ("candidate", "active")

    def mark_used(self, count: int = 1) -> None:
        """Increment used count."""
        self.used_count += count

    def to_dict(self) -> dict:
        """Serialize slot state to dict (tensors -> shape strings)."""
        return {
            "slot_id": self.slot_id,
            "chart_id": self.chart_id,
            "layer_id": self.layer_id,
            "input_rank": self.input_rank,
            "output_rank": self.output_rank,
            "P_shape": list(self.P.shape) if self.P is not None else None,
            "R_shape": list(self.R.shape) if self.R is not None else None,
            "B_shape": list(self.B.shape) if self.B is not None else None,
            "b_shape": list(self.b.shape) if self.b is not None else None,
            "key_shape": list(self.key.shape) if self.key is not None else None,
            "support": self.support,
            "quality": self.quality,
            "state": self.state,
            "created_task_id": self.created_task_id,
            "last_updated_task_id": self.last_updated_task_id,
            "used_count": self.used_count,
        }

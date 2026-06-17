"""ChartState: dataclass for storing per-chart geometry and metadata."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class ChartState:
    """
    State of a single chart in the atlas.

    A chart represents a local region of the feature manifold,
    characterized by a PPCA model (mu, U, eigvals, sigma_perp).
    Charts are task-agnostic: they describe feature geometry only,
    not class labels or task boundaries.

    Attributes:
        chart_id: unique chart identifier within a layer.
        layer_id: ViT block index this chart belongs to.
        mu: mean vector of shape [D].
        U: principal basis of shape [D, rank].
        eigvals: eigenvalues of shape [rank].
        sigma_perp: isotropic noise variance (scalar).
        prior: prior probability p(chart).
        trust_nll: negative log-likelihood at trust_quantile.
        radius_d2: squared Mahalanobis radius covering inliers.
        n_support: number of samples assigned to this chart.
        age: number of tasks this chart has survived.
        hit_count: number of times this chart was selected during routing.
        reuse_count: number of times this chart was reused across tasks.
        state: lifecycle state (candidate/provisional/active/mature/saturated/dormant).
        slot_ids: list of slot ids associated with this chart.
        quality: dict of quality metrics (compactness, stability, etc.).
        created_task_id: task that created this chart.
        last_updated_task_id: last task that updated this chart.
    """

    chart_id: int
    layer_id: int
    mu: Optional[Tensor] = None
    U: Optional[Tensor] = None
    eigvals: Optional[Tensor] = None
    sigma_perp: float = 1.0
    prior: float = 0.0
    trust_nll: float = 0.0
    radius_d2: float = 0.0
    n_support: int = 0
    age: int = 0
    hit_count: int = 0
    # Phase-7.5: shared routing basis
    Q_router: Optional[Tensor] = None       # [D, r_q]
    router_eigvals: Optional[Tensor] = None # [r_q]
    router_rank: int = 0
    reuse_count: int = 0
    state: str = "candidate"
    slot_ids: List[int] = field(default_factory=list)
    quality: Dict[str, float] = field(default_factory=dict)
    created_task_id: int = 0
    last_updated_task_id: int = 0

    def is_active(self) -> bool:
        """Return True if chart is in active or mature state."""
        return self.state in ("active", "mature")

    def is_mature(self) -> bool:
        """Return True if chart has reached mature state."""
        return self.state == "mature"

    def can_update_geometry(self) -> bool:
        """Return True if chart geometry can receive further EMA updates."""
        return self.state in ("candidate", "provisional", "active")

    def can_add_slot(self) -> bool:
        """Return True if new slots can be added to this chart."""
        return self.state in ("active", "mature")

    def mark_hit(self, count: int = 1) -> None:
        """Increment hit count."""
        self.hit_count += count

    def add_slot_id(self, slot_id: int) -> None:
        """Register a slot id on this chart."""
        if slot_id not in self.slot_ids:
            self.slot_ids.append(slot_id)

    def to_dict(self) -> dict:
        """Serialize chart state to dict (tensors -> shape strings)."""
        return {
            "chart_id": self.chart_id,
            "layer_id": self.layer_id,
            "mu_shape": list(self.mu.shape) if self.mu is not None else None,
            "U_shape": list(self.U.shape) if self.U is not None else None,
            "eigvals_shape": list(self.eigvals.shape) if self.eigvals is not None else None,
            "sigma_perp": self.sigma_perp,
            "prior": self.prior,
            "trust_nll": self.trust_nll,
            "radius_d2": self.radius_d2,
            "n_support": self.n_support,
            "age": self.age,
            "hit_count": self.hit_count,
            "reuse_count": self.reuse_count,
            "state": self.state,
            "slot_ids": self.slot_ids,
            "quality": self.quality,
            "created_task_id": self.created_task_id,
            "last_updated_task_id": self.last_updated_task_id,
        }

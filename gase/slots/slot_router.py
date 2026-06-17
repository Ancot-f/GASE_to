"""Slot routers: key-based and teacher-guided slot selection."""

from typing import Dict, List

import torch
from torch import Tensor
from torch import nn

from ..atlas.chart_state import ChartState
from .slot_state import SlotState


class KeyBasedSlotRouter(nn.Module):
    """
    Key-based slot router.

    Selects slots by comparing the projected feature (P^T @ h)
    against stored slot keys via cosine similarity or Euclidean distance.
    No learnable parameters beyond slot keys.
    """

    def __init__(self, dim: int, input_rank: int, top_k: int = 2):
        """
        Args:
            dim: feature dimension D.
            input_rank: rank of input projection.
            top_k: number of top slots to select.
        """
        super().__init__()
        self.dim = dim
        self.input_rank = input_rank
        self.top_k = top_k

    def forward(
        self,
        h_chart: Tensor,
        chart_state: ChartState,
        slot_states: List[SlotState],
    ) -> Tensor:
        """
        Compute slot selection probabilities via key matching.

        Args:
            h_chart: features of shape [B, D].
            chart_state: parent chart (provides P basis).
            slot_states: available slots in this chart.

        Returns:
            Slot probabilities of shape [B, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_slot_scores(
        self,
        h_chart: Tensor,
        chart_state: ChartState,
        slot_states: List[SlotState],
    ) -> Tensor:
        """
        Compute raw slot scores (e.g., negative cosine distance to keys).

        Args:
            h_chart: features [B, D].
            chart_state: parent chart.
            slot_states: available slots.

        Returns:
            Scores of shape [B, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_slot_probs(self, scores: Tensor) -> Tensor:
        """
        Convert scores to probabilities via softmax.

        Args:
            scores: raw scores [B, num_slots].

        Returns:
            Probabilities of shape [B, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")


class TeacherGuidedSlotRouter(nn.Module):
    """
    Teacher-guided slot router.

    Uses a small learnable network that predicts which slots
    best explain the teacher residual for a given feature.

    Trained during the distill phase using teacher-assigned
    soft slot labels.
    """

    def __init__(self, dim: int, input_rank: int, num_slots: int, hidden_dim: int = 64):
        """
        Args:
            dim: feature dimension D.
            input_rank: rank of input projection.
            num_slots: maximum number of slots.
            hidden_dim: hidden dimension of the router MLP.
        """
        super().__init__()
        self.dim = dim
        self.input_rank = input_rank
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim

        self.router = nn.Sequential(
            nn.Linear(dim + input_rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_slots),
        )

    def forward(self, router_features: Tensor) -> Tensor:
        """
        Predict slot probabilities from router features.

        Args:
            router_features: concatenated features of shape [B, dim + input_rank].

        Returns:
            Slot logits of shape [B, num_slots].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def build_router_features(
        self,
        h_chart: Tensor,
        chart_state: ChartState,
        slot_states: List[SlotState],
    ) -> Tensor:
        """
        Build input features for the router.

        Concatenates h_chart with chart-conditional features and
        slot-specific features.

        Args:
            h_chart: features [B, D].
            chart_state: parent chart.
            slot_states: available slots.

        Returns:
            Router input features of shape [B, dim + input_rank].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def update_num_slots(self, num_slots: int) -> None:
        """
        Update the output dimension when slots are added/removed.

        Args:
            num_slots: new number of slots.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

"""ChartAdapter: chart-slot-bound low-rank residual transformations.

Formula:
    A_{c,s}(h) = b_{c,s} + R_{c,s} * f_{c,s}(P_{c,s}^T * (h - mu_c))

where:
    P in R^{D x input_rank}: input projection to chart tangent space.
    R in R^{output_rank x D}: output projection back to feature space.
    f_{c,s}: learned transformation (linear or MLP) in the low-rank latent space.
    b_{c,s}: residual bias.
    h: pre-adapter feature.
    mu_c: chart mean.
"""

import torch
from torch import Tensor
from torch import nn


class LinearChartAdapter(nn.Module):
    """
    Linear chart-slot adapter.

    Performs: delta = b + R @ B @ P^T @ (h - mu)

    where B is a learned linear map in the low-rank latent space.
    """

    def __init__(
        self,
        dim: int,
        input_rank: int = 8,
        output_rank: int = 4,
        chart_id: int = -1,
        slot_id: int = -1,
        layer_id: int = -1,
    ):
        """
        Args:
            dim: feature dimension D.
            input_rank: rank of input projection P (D -> input_rank).
            output_rank: rank of output projection R (output_rank -> D).
            chart_id: chart this adapter belongs to.
            slot_id: slot this adapter belongs to.
            layer_id: ViT block index.
        """
        super().__init__()
        self.dim = dim
        self.input_rank = input_rank
        self.output_rank = output_rank
        self.chart_id = chart_id
        self.slot_id = slot_id
        self.layer_id = layer_id

        # Projection bases (set externally after distillation)
        self.register_buffer("P", torch.empty(dim, input_rank))
        self.register_buffer("R", torch.empty(output_rank, dim))
        # Linear map in latent space
        self.B = nn.Parameter(torch.zeros(output_rank, input_rank))
        # Residual bias
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, h_chart: Tensor, mu: Tensor) -> Tensor:
        """
        Compute chart-adapter residual.

        Args:
            h_chart: pre-adapter features of shape [B, D].
            mu: chart mean of shape [D].

        Returns:
            Residual delta_chart of shape [B, D].
        """
        # h_centered: [B, D]
        h_centered = h_chart - mu.unsqueeze(0)
        # z_in: [B, input_rank]
        z_in = h_centered @ self.P
        # z_out: [B, output_rank]
        z_out = z_in @ self.B.T
        # delta: [B, D]
        delta = z_out @ self.R + self.b.unsqueeze(0)
        return delta

    def set_projection_bases(self, P: Tensor, R: Tensor) -> None:
        """Set the projection bases P and R (from PPCA + slot basis)."""
        self.P = P.to(self.P.device)
        self.R = R.to(self.R.device)

    def set_linear_map(self, B: Tensor, b: Tensor) -> None:
        """Set the learned linear map B and bias b."""
        self.B.data = B.to(self.B.device)
        self.b.data = b.to(self.b.device)

    def extra_repr(self) -> str:
        return (
            f"chart_id={self.chart_id}, slot_id={self.slot_id}, "
            f"layer_id={self.layer_id}, input_rank={self.input_rank}, "
            f"output_rank={self.output_rank}"
        )


class MLPChartAdapter(nn.Module):
    """
    Non-linear chart-slot adapter with an internal MLP.

    Performs: delta = b + R @ MLP(P^T @ (h - mu))

    The MLP operates in the low-rank input space.
    """

    def __init__(
        self,
        dim: int,
        input_rank: int = 8,
        output_rank: int = 4,
        hidden_dim: int = 8,
        chart_id: int = -1,
        slot_id: int = -1,
        layer_id: int = -1,
    ):
        """
        Args:
            dim: feature dimension D.
            input_rank: rank of input projection P.
            output_rank: rank of output projection R.
            hidden_dim: hidden dimension of the internal MLP.
            chart_id: chart this adapter belongs to.
            slot_id: slot this adapter belongs to.
            layer_id: ViT block index.
        """
        super().__init__()
        self.dim = dim
        self.input_rank = input_rank
        self.output_rank = output_rank
        self.hidden_dim = hidden_dim
        self.chart_id = chart_id
        self.slot_id = slot_id
        self.layer_id = layer_id

        # Projection bases
        self.register_buffer("P", torch.empty(dim, input_rank))
        self.register_buffer("R", torch.empty(output_rank, dim))
        # MLP in latent space
        self.mlp = nn.Sequential(
            nn.Linear(input_rank, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_rank),
        )
        # Residual bias
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, h_chart: Tensor, mu: Tensor) -> Tensor:
        """
        Compute chart-adapter residual with non-linear MLP.

        Args:
            h_chart: pre-adapter features of shape [B, D].
            mu: chart mean of shape [D].

        Returns:
            Residual delta_chart of shape [B, D].
        """
        h_centered = h_chart - mu.unsqueeze(0)
        z_in = h_centered @ self.P  # [B, input_rank]
        z_out = self.mlp(z_in)      # [B, output_rank]
        delta = z_out @ self.R + self.b.unsqueeze(0)
        return delta

    def set_projection_bases(self, P: Tensor, R: Tensor) -> None:
        """Set the projection bases P and R."""
        self.P = P.to(self.P.device)
        self.R = R.to(self.R.device)

    def set_residual_bias(self, b: Tensor) -> None:
        """Set the residual bias b."""
        self.b.data = b.to(self.b.device)

    def extra_repr(self) -> str:
        return (
            f"chart_id={self.chart_id}, slot_id={self.slot_id}, "
            f"layer_id={self.layer_id}, input_rank={self.input_rank}, "
            f"output_rank={self.output_rank}, hidden_dim={self.hidden_dim}"
        )

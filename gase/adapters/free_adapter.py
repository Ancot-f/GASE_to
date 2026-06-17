"""FreeAdapter: absorbs residual leftover unexplained by chart/slot adapters."""

import torch
from torch import Tensor
from torch import nn


class FreeAdapter(nn.Module):
    """
    Free adapter for absorbing residual leftover.

    Architecture: D -> bottleneck -> D, with a learnable gate.
    The gate controls how much free-adapter contribution enters
    the final residual combination.

    Attributes:
        dim: feature dimension D.
        bottleneck_dim: hidden bottleneck dimension.
        dropout: dropout rate.
        scale: output scaling factor.
        gate: learnable scalar gate for mixing.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 16,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        """
        Args:
            dim: feature dimension D.
            bottleneck_dim: hidden bottleneck dimension.
            dropout: dropout rate.
            scale: output scaling factor.
        """
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale

        self.gate = nn.Parameter(torch.tensor(0.0))
        self.down_proj = nn.Linear(dim, bottleneck_dim, bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up_proj = nn.Linear(bottleneck_dim, dim, bias=True)

        # SEMA-identical init (init_option="lora")
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, h_chart: Tensor) -> Tensor:
        """
        Compute free-adapter residual.

        Args:
            h_chart: pre-adapter features of shape [B, D].

        Returns:
            Residual delta_free of shape [B, D].
        """
        delta = self.down_proj(h_chart)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up_proj(delta)
        return delta * self.scale * torch.sigmoid(self.gate)

    def set_gate(self, value: float) -> None:
        """Manually set the gate parameter."""
        with torch.no_grad():
            self.gate.fill_(value)

    def reset_parameters(self) -> None:
        """Reinitialize adapter parameters."""
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.gate)

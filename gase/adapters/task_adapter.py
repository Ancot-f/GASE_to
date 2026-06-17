"""TaskAdapter: per-task temporary teacher adapter (MLP bottleneck)."""

import torch
from torch import Tensor
from torch import nn


class TaskAdapter(nn.Module):
    """
    Per-task adapter trained temporarily to capture task-specific residuals.

    Architecture: D -> bottleneck -> D, with optional dropout and scaling.
    Trained only on the current task and discarded after distillation.

    Attributes:
        dim: feature dimension D.
        bottleneck_dim: hidden bottleneck dimension.
        dropout: dropout rate.
        scale: output scaling factor.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 16,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale

        self.down_proj = nn.Linear(dim, bottleneck_dim, bias=True)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up_proj = nn.Linear(bottleneck_dim, dim, bias=True)

        # SEMA-identical init (init_option="lora")
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, h: Tensor) -> Tensor:
        """
        Compute task-adapter residual.

        Args:
            h: input features of shape [B, D].

        Returns:
            Residual delta of shape [B, D].
        """
        delta = self.down_proj(h)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up_proj(delta)
        return delta * self.scale

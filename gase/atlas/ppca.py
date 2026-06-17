"""PPCAEstimator: Probabilistic PCA for chart geometry modeling."""

import torch
from torch import Tensor
from torch import nn

from .chart_state import ChartState


class PPCAEstimator(nn.Module):
    """
    Probabilistic PCA estimator for chart-local feature modeling.

    Models h = mu + U * z + epsilon, where z ~ N(0, diag(eigvals))
    and epsilon ~ N(0, sigma_perp * I).

    This provides a low-rank Gaussian approximation of the
    feature distribution within a chart.
    """

    def __init__(self, dim: int, rank: int):
        """
        Args:
            dim: feature dimension D.
            rank: number of principal components (chart rank).
        """
        super().__init__()
        self.dim = dim
        self.rank = rank

    def fit(self, h_chart: Tensor, rank: int) -> None:
        """
        Fit PPCA parameters to data via EM or analytic solution.

        Args:
            h_chart: features of shape [N, D].
            rank: rank to use (may differ from initialized rank).
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def transform(self, h_chart: Tensor) -> Tensor:
        """
        Project features to latent space: z = U^T (h - mu).

        Args:
            h_chart: features of shape [B, D].

        Returns:
            Latent codes of shape [B, rank].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def inverse_transform(self, z: Tensor) -> Tensor:
        """
        Reconstruct features from latent codes: h_hat = mu + U * z.

        Args:
            z: latent codes of shape [B, rank].

        Returns:
            Reconstructed features of shape [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def nll(self, h_chart: Tensor) -> Tensor:
        """
        Compute per-sample negative log-likelihood under PPCA model.

        Args:
            h_chart: features of shape [B, D].

        Returns:
            NLL values of shape [B].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def to_chart_state(
        self,
        layer_id: int,
        chart_id: int,
    ) -> ChartState:
        """
        Export current PPCA parameters to a ChartState.

        Args:
            layer_id: ViT block index.
            chart_id: unique chart id.

        Returns:
            ChartState with mu, U, eigvals, sigma_perp populated.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

"""PPCAEstimator: Probabilistic PCA for chart geometry modeling."""

import torch
from torch import Tensor
from torch import nn

from .chart_state import ChartState
from ..geometry.pca import compute_pca_basis, project_to_basis, reconstruct_from_basis


class PPCAEstimator(nn.Module):
    """
    Probabilistic PCA estimator for chart-local feature modeling.

    Models h = mu + U * z + epsilon, where z ~ N(0, diag(eigvals))
    and epsilon ~ N(0, sigma_perp * I).
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
        self.mu: Tensor = torch.zeros(dim)
        self.U: Tensor = torch.zeros(dim, rank)
        self.eigvals: Tensor = torch.zeros(rank)
        self.sigma_perp: float = 1.0
        self.radius_d2: float = 0.0
        self.n_support: int = 0

    def fit(self, h_chart: Tensor, rank: int) -> "PPCAEstimator":
        """
        Fit PPCA parameters via PCA + residual variance estimation.

        Args:
            h_chart: features of shape [N, D].
            rank: PCA rank to use.

        Returns:
            self
        """
        if h_chart.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples, got {h_chart.shape[0]}")

        actual_rank = min(rank, h_chart.shape[0] - 1, self.dim)
        self.rank = actual_rank

        self.mu, self.U, self.eigvals = compute_pca_basis(h_chart, actual_rank)
        self.n_support = h_chart.shape[0]

        # Compute sigma_perp from reconstruction residual
        z = project_to_basis(h_chart, self.U, self.mu)
        h_rec = reconstruct_from_basis(z, self.U, self.mu)
        residuals = h_chart - h_rec
        self.sigma_perp = float(residuals.pow(2).mean().sqrt())

        # Compute radius_d2: PPCA Mahalanobis distance at 95th percentile
        with torch.no_grad():
            d2 = self._ppca_mahalanobis_d2(h_chart)
            self.radius_d2 = float(torch.quantile(d2, 0.95))

        return self

    def _ppca_mahalanobis_d2(self, h_chart: Tensor) -> Tensor:
        """Compute PPCA Mahalanobis d^2 per sample."""
        h_centered = h_chart - self.mu.unsqueeze(0)
        z = h_centered @ self.U  # [N, rank]
        tangent_term = (z ** 2 / (self.eigvals.unsqueeze(0) + 1e-8)).sum(dim=1)
        normal_residual = h_centered - z @ self.U.mT
        normal_term = (normal_residual ** 2).sum(dim=1) / max(self.sigma_perp ** 2, 1e-8)
        return tangent_term + normal_term

    def transform(self, h_chart: Tensor) -> Tensor:
        """
        Project features to latent space: z = U^T (h - mu).

        Args:
            h_chart: features of shape [B, D].

        Returns:
            Latent codes of shape [B, rank].
        """
        return project_to_basis(h_chart, self.U, self.mu)

    def inverse_transform(self, z: Tensor) -> Tensor:
        """
        Reconstruct features from latent codes: h_hat = mu + U * z.

        Args:
            z: latent codes of shape [B, rank].

        Returns:
            Reconstructed features of shape [B, D].
        """
        return reconstruct_from_basis(z, self.U, self.mu)

    def nll(self, h_chart: Tensor) -> Tensor:
        """Per-sample negative log-likelihood (placeholder — Phase-5+)."""
        raise NotImplementedError("Phase-4 does not implement NLL.")

    def to_chart_state(self, layer_id: int, chart_id: int) -> ChartState:
        """
        Export current PPCA parameters to a ChartState.

        Args:
            layer_id: ViT block index.
            chart_id: unique chart id.

        Returns:
            ChartState with mu, U, eigvals, sigma_perp populated.
        """
        return ChartState(
            chart_id=chart_id,
            layer_id=layer_id,
            mu=self.mu.clone().detach(),
            U=self.U.clone().detach(),
            eigvals=self.eigvals.clone().detach(),
            sigma_perp=self.sigma_perp,
            prior=1.0,
            trust_nll=0.0,
            radius_d2=self.radius_d2,
            n_support=self.n_support,
            age=0,
            hit_count=0,
            reuse_count=0,
            state="active",
            slot_ids=[],
            quality={},
            created_task_id=-1,
            last_updated_task_id=-1,
        )

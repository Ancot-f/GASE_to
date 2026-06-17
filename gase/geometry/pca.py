"""PCA utilities: basis computation, projection, reconstruction."""

from typing import Tuple

import torch
from torch import Tensor


def compute_pca_basis(x: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute PCA mean, principal basis, and eigenvalues via SVD.

    Args:
        x: data matrix of shape [N, D].
        rank: number of principal components (must be <= min(N-1, D)).

    Returns:
        Tuple of (mean [D], basis [D, rank], eigenvalues [rank]).
    """
    N, D = x.shape
    if rank > min(N - 1, D):
        raise ValueError(f"rank {rank} > min(N-1={N-1}, D={D}) = {min(N-1, D)}")

    mean = x.mean(dim=0)
    x_centered = x - mean.unsqueeze(0)  # [N, D]

    # Use truncated SVD for efficiency when D is large
    if D > 2 * rank:
        # Use torch.pca_lowrank for memory efficiency
        U_s, S, V = torch.pca_lowrank(x_centered, q=rank)
    else:
        U_s, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
        V = Vh.mT[:, :rank]
        S = S[:rank]

    basis = V[:, :rank]  # [D, rank]
    eigvals = S[:rank] ** 2 / max(N - 1, 1)

    return mean, basis.to(x.dtype), eigvals.to(x.dtype)


def compute_low_rank_svd(x: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute truncated SVD of data matrix.

    Args:
        x: data matrix of shape [N, D].
        rank: number of singular vectors.

    Returns:
        Tuple of (U [N, rank], S [rank], V [D, rank]).
    """
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return U[:, :rank], S[:rank], Vh.mT[:, :rank]


def project_to_basis(x: Tensor, basis: Tensor, mean: Tensor) -> Tensor:
    """
    Project data onto a given basis.

    z = basis^T @ (x - mean)

    Args:
        x: data of shape [N, D].
        basis: projection basis of shape [D, rank].
        mean: mean vector of shape [D].

    Returns:
        Latent codes of shape [N, rank].
    """
    return (x - mean.unsqueeze(0)) @ basis


def reconstruct_from_basis(z: Tensor, basis: Tensor, mean: Tensor) -> Tensor:
    """
    Reconstruct data from latent codes.

    x_hat = mean + basis @ z

    Args:
        z: latent codes of shape [N, rank].
        basis: projection basis of shape [D, rank].
        mean: mean vector of shape [D].

    Returns:
        Reconstructed data of shape [N, D].
    """
    return mean.unsqueeze(0) + z @ basis.mT

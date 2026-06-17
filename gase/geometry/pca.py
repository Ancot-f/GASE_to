"""PCA utilities: basis computation, projection, reconstruction."""

from typing import Tuple

from torch import Tensor


def compute_pca_basis(x: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute PCA basis via SVD of centered data.

    Args:
        x: data matrix of shape [N, D].
        rank: number of principal components.

    Returns:
        Tuple of (mean [D], basis [D, rank], eigenvalues [rank]).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_low_rank_svd(x: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute truncated SVD of data matrix.

    Args:
        x: data matrix of shape [N, D].
        rank: number of singular vectors.

    Returns:
        Tuple of (U [N, rank], S [rank], V [D, rank]).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def project_to_basis(x: Tensor, basis: Tensor) -> Tensor:
    """
    Project data onto a given basis.

    z = basis^T @ (x - mean)

    Args:
        x: data of shape [N, D].
        basis: projection basis of shape [D, rank].

    Returns:
        Latent codes of shape [N, rank].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


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
    raise NotImplementedError("Phase-0 skeleton only.")

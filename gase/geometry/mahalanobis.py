"""Mahalanobis distance utilities for inlier/outlier detection."""

from torch import Tensor


def mahalanobis_distance(
    h: Tensor,
    mu: Tensor,
    cov_inv: Tensor,
) -> Tensor:
    """
    Compute squared Mahalanobis distance.

    d^2 = (h - mu)^T cov_inv (h - mu)

    Args:
        h: features of shape [B, D].
        mu: mean of shape [D].
        cov_inv: inverse covariance of shape [D, D].

    Returns:
        Squared distances of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def ppca_mahalanobis_distance(
    h: Tensor,
    mu: Tensor,
    U: Tensor,
    eigvals: Tensor,
    sigma_perp: float,
) -> Tensor:
    """
    Compute PPCA-based Mahalanobis distance.

    Uses the Woodbury identity for efficient computation:
    d^2 = (1/sigma_perp) * (||h - mu||^2 - ||U^T(h-mu)||^2_M)

    Args:
        h: features of shape [B, D].
        mu: mean of shape [D].
        U: basis of shape [D, rank].
        eigvals: eigenvalues of shape [rank].
        sigma_perp: isotropic noise variance.

    Returns:
        Squared PPCA Mahalanobis distances of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def normal_residual_distance(
    h: Tensor,
    mu: Tensor,
    U: Tensor,
    sigma_perp: float,
) -> Tensor:
    """
    Compute normal-space residual distance.

    d_normal^2 = ||(I - U U^T)(h - mu)||^2 / sigma_perp

    This measures how far a point is from the chart's tangent space.

    Args:
        h: features of shape [B, D].
        mu: mean of shape [D].
        U: basis of shape [D, rank].
        sigma_perp: normal variance.

    Returns:
        Normal residual distances of shape [B].
    """
    raise NotImplementedError("Phase-0 skeleton only.")

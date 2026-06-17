"""Grassmann manifold utilities for subspace comparison and interpolation."""

from torch import Tensor


def grassmann_distance(U1: Tensor, U2: Tensor) -> Tensor:
    """
    Compute Grassmann distance between two subspaces.

    d(U1, U2) = ||theta||_2 where theta are the principal angles.

    Args:
        U1: first basis of shape [D, rank].
        U2: second basis of shape [D, rank].

    Returns:
        Scalar Grassmann distance.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def principal_angles(U1: Tensor, U2: Tensor) -> Tensor:
    """
    Compute principal angles between two subspaces.

    theta_i = arccos(sigma_i) where sigma_i are singular values of U1^T U2.

    Args:
        U1: first basis of shape [D, rank].
        U2: second basis of shape [D, rank].

    Returns:
        Principal angles of shape [rank] in radians.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def subspace_similarity(U1: Tensor, U2: Tensor) -> Tensor:
    """
    Compute subspace similarity via average cosine of principal angles.

    sim = mean(cos(theta_i)) = mean(sigma_i)

    Args:
        U1: first basis [D, rank].
        U2: second basis [D, rank].

    Returns:
        Scalar similarity in [0, 1].
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def grassmann_ema_update(
    U_old: Tensor,
    U_new: Tensor,
    momentum: float,
) -> Tensor:
    """
    EMA update of a subspace on the Grassmann manifold.

    Interpolates between U_old and U_new along the Grassmann geodesic.
    U_updated = U_old * cos(m * theta) + U_perp * sin(m * theta)

    Args:
        U_old: current basis [D, rank].
        U_new: new data basis [D, rank].
        momentum: interpolation factor in [0, 1].

    Returns:
        Updated basis [D, rank].
    """
    raise NotImplementedError("Phase-0 skeleton only.")

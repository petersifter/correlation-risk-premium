"""Average correlation from index and constituent volatilities.

With basket weights w_i summing to 1, constituent volatilities sigma_i, pairwise
correlations rho_ij and index volatility sigma_I:

    sigma_I^2 = sum_i sum_j w_i w_j rho_ij sigma_i sigma_j                      (1)

Replace every off-diagonal rho_ij with a single rho_bar and solve. Writing
A = sum_i w_i sigma_i and B = sum_i w_i^2 sigma_i^2, the off-diagonal sum
sum_{i != j} w_i w_j sigma_i sigma_j equals A^2 - B, so

    rho_bar = (sigma_I^2 - B) / (A^2 - B)                                       (2)

Implied volatilities give implied correlation, realized volatilities give
realized correlation. (2) is exact when all rho_ij are equal. At rho_bar = 1 the
index is as volatile as its weighted-average constituent (sigma_I = A); at
rho_bar = 0, sigma_I = sqrt(B).

Weights must sum to 1. A top-50 basket is renormalized by the caller.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_WEIGHT_TOLERANCE = 1e-8


def _validate(weights: ArrayLike, component_vols: ArrayLike) -> tuple[NDArray, NDArray]:
    """Check weights and volatilities and return them as float arrays."""
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(component_vols, dtype=float)

    if w.ndim != 1 or sigma.ndim != 1:
        raise ValueError("weights and component_vols must both be one-dimensional")
    if w.shape != sigma.shape:
        raise ValueError(f"weights has length {w.size} but component_vols has length {sigma.size}")
    if w.size < 2:
        raise ValueError("need at least two constituents")
    if np.any(sigma < 0):
        raise ValueError("component_vols must be non-negative")
    if abs(w.sum() - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(f"weights must sum to 1, got {w.sum():.10g}")
    return w, sigma


def index_variance(
    component_vols: ArrayLike,
    weights: ArrayLike,
    correlation_matrix: ArrayLike,
) -> float:
    """Index variance from the full correlation matrix, equation (1): w' C w."""
    w, sigma = _validate(weights, component_vols)
    rho = np.asarray(correlation_matrix, dtype=float)

    n = w.size
    if rho.shape != (n, n):
        raise ValueError(f"correlation_matrix must be {n}x{n}, got {rho.shape}")
    if not np.allclose(rho, rho.T):
        raise ValueError("correlation_matrix must be symmetric")
    if not np.allclose(np.diag(rho), 1.0):
        raise ValueError("correlation_matrix must have unit diagonal")

    covariance = rho * np.outer(sigma, sigma)
    return float(w @ covariance @ w)


def index_vol_from_average_correlation(
    component_vols: ArrayLike,
    weights: ArrayLike,
    rho_bar: float,
) -> float:
    """sigma_I = sqrt(B + rho_bar (A^2 - B)), the inverse of `average_correlation`."""
    w, sigma = _validate(weights, component_vols)

    weighted_avg_vol = float(w @ sigma)
    diagonal = float(np.sum(w**2 * sigma**2))
    variance = diagonal + rho_bar * (weighted_avg_vol**2 - diagonal)

    if variance < 0.0:
        raise ValueError(f"rho_bar={rho_bar:.6g} implies a negative index variance")
    return float(np.sqrt(variance))


def average_correlation(
    index_vol: float,
    component_vols: ArrayLike,
    weights: ArrayLike,
) -> float:
    """rho_bar = (sigma_I^2 - B) / (A^2 - B), equation (2).

    Not clipped to [-1, 1]. A value outside that range means the index and
    constituent volatilities don't fit together (mismatched dates, a stale
    quote, a basket that doesn't track the index), and clipping would hide it.
    """
    w, sigma = _validate(weights, component_vols)

    weighted_avg_vol = float(w @ sigma)
    diagonal = float(np.sum(w**2 * sigma**2))
    off_diagonal = weighted_avg_vol**2 - diagonal

    if off_diagonal <= 0.0:
        raise ValueError("A^2 - B is not positive, so average correlation is undefined")
    return (index_vol**2 - diagonal) / off_diagonal

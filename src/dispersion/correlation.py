"""Correlation algebra for dispersion trading.

Dispersion is a bet on the gap between the correlation priced into index options and the
correlation the constituents actually realise. Everything in this module follows from one exact
identity and one deliberate approximation, both stated below.

Notation
--------
``w_i``      weight of constituent *i*, with ``sum_i w_i = 1``
``sigma_i``  volatility of constituent *i*
``rho_ij``   correlation between constituents *i* and *j*
``sigma_I``  volatility of the index

The exact identity
------------------
The index return is ``R_I = sum_i w_i R_i``, so its variance is a weighted double sum over the
covariance matrix::

    sigma_I^2 = sum_i sum_j w_i w_j rho_ij sigma_i sigma_j                              (1)

Splitting the diagonal (``i == j``, where ``rho_ii = 1``) from the off-diagonal terms::

    sigma_I^2 = sum_i w_i^2 sigma_i^2 + sum_{i != j} w_i w_j rho_ij sigma_i sigma_j     (2)

Both lines are exact. No assumption has been made yet.

The approximation
-----------------
There are ``n (n - 1) / 2`` distinct pairwise correlations - about 125,000 for the S&P 500 - and
the market quotes none of them. So replace every off-diagonal ``rho_ij`` with a single average
correlation ``rho_bar`` and solve (2) for it::

    rho_bar = (sigma_I^2 - sum_i w_i^2 sigma_i^2) / (sum_{i != j} w_i w_j sigma_i sigma_j)

Writing ``A = sum_i w_i sigma_i`` for the weighted-average volatility and
``B = sum_i w_i^2 sigma_i^2`` for the diagonal term, the off-diagonal sum has a closed form,
``sum_{i != j} w_i w_j sigma_i sigma_j = A^2 - B``, so::

    rho_bar = (sigma_I^2 - B) / (A^2 - B)                                               (3)

Equation (3) is what the market means by *implied correlation* when the volatilities are implied,
and what we mean by *realised correlation* when they are realised. The dispersion trade is short
the first and long the second.

This is the approximation used by Cboe's implied correlation indices and by every dispersion desk;
it is not a shortcut invented here. Note that it is **exact**, not approximate, whenever the
pairwise correlations are homogeneous - the error comes entirely from dispersion in the ``rho_ij``
themselves. ``tests/test_correlation.py`` asserts exactly that.

Two limiting cases, worth carrying in your head as a permanent sanity check::

    rho_bar = 1  =>  sigma_I = A          no diversification: the index is exactly as volatile as
                                          its weighted-average constituent
    rho_bar = 0  =>  sigma_I = sqrt(B)    independent names: the index is far quieter than anything
                                          in it

Everything here is computed in *variance*, never volatility, because variance is the quantity that
adds. Averaging volatilities is meaningless.

A note on weights
-----------------
Every function requires ``sum_i w_i == 1``. Real dispersion books trade a subset - the largest 50
names rather than all 500 - whose index weights sum to well under one. The correct handling is to
**renormalise the subset weights to sum to one** and treat the result as a proxy index, which is
what desks do in practice. That renormalisation is deliberately the caller's job, so the choice
stays visible at the call site instead of being buried in here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "average_correlation",
    "index_variance",
    "index_vol_from_average_correlation",
]

_WEIGHT_TOLERANCE = 1e-8


def _validate(weights: ArrayLike, component_vols: ArrayLike) -> tuple[NDArray, NDArray]:
    """Coerce and check weights and volatilities, returning them as float arrays."""
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(component_vols, dtype=float)

    if w.ndim != 1 or sigma.ndim != 1:
        raise ValueError("weights and component_vols must both be one-dimensional")
    if w.shape != sigma.shape:
        raise ValueError(f"weights has length {w.size} but component_vols has length {sigma.size}")
    if w.size < 2:
        raise ValueError("need at least two constituents; correlation is undefined for one")
    if np.any(sigma < 0):
        raise ValueError("component_vols must be non-negative")
    if abs(w.sum() - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(f"weights must sum to 1, got {w.sum():.10g}; renormalise before calling")
    return w, sigma


def index_variance(
    component_vols: ArrayLike,
    weights: ArrayLike,
    correlation_matrix: ArrayLike,
) -> float:
    """Index variance from the full correlation matrix, equation (1). Exact.

    ``sigma_I^2 = w' C w`` where ``C_ij = rho_ij sigma_i sigma_j`` is the covariance matrix.

    This is the ground truth that the single-correlation approximation is measured against.
    """
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
    """Index volatility implied by a single average correlation, equation (3) rearranged.

    ``sigma_I = sqrt(B + rho_bar (A^2 - B))`` with ``A = sum w_i sigma_i`` and
    ``B = sum w_i^2 sigma_i^2``.

    The inverse of :func:`average_correlation`, and the function that produces the two limiting
    cases: ``rho_bar = 1`` returns ``A``, ``rho_bar = 0`` returns ``sqrt(B)``.
    """
    w, sigma = _validate(weights, component_vols)

    weighted_avg_vol = float(w @ sigma)
    diagonal = float(np.sum(w**2 * sigma**2))
    variance = diagonal + rho_bar * (weighted_avg_vol**2 - diagonal)

    if variance < 0.0:
        raise ValueError(
            f"rho_bar={rho_bar:.6g} implies a negative index variance; no covariance matrix with "
            "this average correlation exists for these weights and volatilities"
        )
    return float(np.sqrt(variance))


def average_correlation(
    index_vol: float,
    component_vols: ArrayLike,
    weights: ArrayLike,
) -> float:
    """Average correlation backed out of an index volatility, equation (3).

    ``rho_bar = (sigma_I^2 - B) / (A^2 - B)`` with ``A = sum w_i sigma_i`` and
    ``B = sum w_i^2 sigma_i^2``.

    Pass implied volatilities to get implied correlation - the market's price for correlation.
    Pass realised volatilities to get realised correlation. The dispersion premium is the gap
    between the two.

    The result is deliberately not clipped to ``[-1, 1]``. A value outside that range is real
    information: it means the index volatility supplied is inconsistent with the constituent
    volatilities, which in live data almost always indicates mismatched observation dates, a stale
    quote, or a constituent set that does not actually span the index. Clipping would hide a data
    bug, so it is left to the caller to notice.
    """
    w, sigma = _validate(weights, component_vols)

    weighted_avg_vol = float(w @ sigma)
    diagonal = float(np.sum(w**2 * sigma**2))
    off_diagonal = weighted_avg_vol**2 - diagonal

    if off_diagonal <= 0.0:
        raise ValueError(
            "the off-diagonal term A^2 - B is not positive, so average correlation is undefined; "
            "this happens when all weight sits on one name or every volatility is zero"
        )
    return (index_vol**2 - diagonal) / off_diagonal

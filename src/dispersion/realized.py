"""Realised volatility and realised correlation from panels of daily returns.

Rung 2. Rung 1 gave the algebra relating index volatility, constituent volatilities and average
correlation. This module applies it to a time series, so that the output is a rolling history of
how correlated the index members actually were.

Nothing here knows where the returns came from. Fetching them from CRSP is rung 2c and lives in
``data.py``; keeping the arithmetic separate from the data access means this layer can be tested
exhaustively on synthetic panels, and the tests pass on a clean clone with no WRDS credentials.

Why returns are not demeaned
----------------------------
``realized_vol`` defaults to ``demean=False``, computing ``sqrt(mean(r^2))`` rather than the sample
standard deviation. Two reasons, and the second is the important one:

1. Over a 21-day window the mean daily return is almost entirely noise. Subtracting an estimate of
   something that small adds variance to the estimator rather than removing bias.
2. Implied volatility is a pure second-moment quantity - an option price says nothing about drift.
   Rung 4 compares realised against implied, and that comparison is only meaningful if both sides
   measure the same thing. Demeaning one side and not the other is a silent mismatch.

This is the standard convention for realised volatility in the options literature, and it is what
variance swaps actually pay on.

The measurement decision
------------------------
There are two ways to obtain the index return series, and they answer different questions.

1. **Construct it** from the constituents, ``R_I = sum_i w_i R_i`` (:func:`basket_return`). Then
   rung 1's identity holds *by construction*, and the measured average correlation is exactly the
   weighted mean of the realised pairwise correlations. This proves the implementation is right,
   but it is circular as a measurement - it measures our own arithmetic.
2. **Use the actual index return.** Now the measurement is real, but our basket does not exactly
   reproduce the index: possibly a subset of names, cap weights that approximate the official
   float-adjusted ones, different corporate-action handling. The difference between the two series
   is a basis we are obliged to measure and report, not hide.

Rung 3 compares implied correlation taken from *SPX options*, so the index leg must be the real
index. Option 2 is therefore the measurement and option 1 is the test; ``tests/test_realized.py``
exercises both.

Missing data
------------
Every function here requires complete panels and raises on ``NaN``. Real CRSP panels are full of
holes - names listing, delisting, halting - and deciding what to do about each hole is a research
decision with consequences for survivorship bias. That decision belongs at the data layer where
the holes are visible, not buried in an arithmetic helper that would quietly drop names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "average_correlation_panel",
    "basket_return",
    "realized_vol",
]

TRADING_DAYS_PER_YEAR = 252
_WEIGHT_TOLERANCE = 1e-6


def _require_complete(frame: pd.DataFrame | pd.Series, name: str) -> None:
    """Reject NaN. See the module docstring on why missingness is the data layer's problem."""
    if frame.isna().to_numpy().any():
        raise ValueError(
            f"{name} contains NaN; resolve missing observations at the data layer, where the "
            "survivorship consequences of dropping or filling a name are visible"
        )


def realized_vol(
    returns: pd.DataFrame | pd.Series,
    window: int,
    *,
    demean: bool = False,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame | pd.Series:
    """Rolling annualised realised volatility.

    With ``demean=False`` (the default, and the options-market convention)::

        sigma_t = sqrt( mean(r_s^2 for s in window ending at t) * trading_days )

    With ``demean=True`` the sample standard deviation is used instead. See the module docstring for
    why the default does not demean - the short version is that rung 4 compares this against
    implied volatility, which is a pure second-moment quantity.

    The window is right-closed and trailing, so ``sigma_t`` uses returns up to and including *t*
    and nothing after it. The first ``window - 1`` rows are NaN.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    _require_complete(returns, "returns")

    if demean:
        variance = returns.rolling(window).var(ddof=1)
    else:
        variance = (returns**2).rolling(window).mean()

    return np.sqrt(variance * trading_days)


def basket_return(
    component_returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """Return of the weighted basket, ``R_I = sum_i w_i R_i``.

    Used to build the *constructed* index series described in the module docstring - the one that
    makes rung 1's identity exact and so serves as the test rather than the measurement.

    ``weights`` must share the index and columns of ``component_returns``, and each row must sum
    to 1.
    """
    _validate_panel(component_returns, weights)
    return (component_returns * weights).sum(axis=1)


def average_correlation_panel(
    index_vol: pd.Series,
    component_vols: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """Rolling average correlation, rung 1's equation (3) applied row by row.

    ``rho_bar_t = (sigma_I_t^2 - B_t) / (A_t^2 - B_t)`` with ``A_t = sum_i w_it sigma_it`` and
    ``B_t = sum_i w_it^2 sigma_it^2``.

    This is a vectorised reimplementation of :func:`dispersion.correlation.average_correlation`
    rather than a loop over dates, because a Python loop over several thousand dates to apply four
    arithmetic operations is not worth writing. The test suite asserts the two agree row by row, so
    the scalar function remains the reference implementation and this one remains honest.

    Pass realised volatilities to get realised correlation. Rung 3 passes implied volatilities to
    the same function to get implied correlation; the arithmetic does not care which, and the fact
    that it does not care is the reason the two are comparable at all.

    Rows where any input is NaN - the leading ``window - 1`` rows of a rolling volatility, for
    instance - come back as NaN rather than raising, since that is expected rather than a data
    problem. Rows whose weights do not sum to 1 are rejected.
    """
    if not isinstance(index_vol, pd.Series):
        raise TypeError("index_vol must be a Series indexed by date")
    _validate_panel(component_vols, weights, allow_nan=True)

    if not index_vol.index.equals(component_vols.index):
        raise ValueError("index_vol and component_vols must share the same index")

    weighted_avg_vol = (weights * component_vols).sum(axis=1, skipna=False)
    diagonal = (weights**2 * component_vols**2).sum(axis=1, skipna=False)
    off_diagonal = weighted_avg_vol**2 - diagonal

    # A^2 - B is the sum of w_i w_j sigma_i sigma_j over i != j, which is positive whenever at
    # least two names carry weight and volatility. Zero means a degenerate basket, not a
    # correlation of infinity.
    off_diagonal = off_diagonal.where(off_diagonal > 0.0)

    return (index_vol**2 - diagonal) / off_diagonal


def _validate_panel(
    component_frame: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    allow_nan: bool = False,
) -> None:
    """Check that a component panel and its weights are aligned and that the weights are weights."""
    if not isinstance(component_frame, pd.DataFrame) or not isinstance(weights, pd.DataFrame):
        raise TypeError("component panel and weights must both be DataFrames")
    if not component_frame.index.equals(weights.index):
        raise ValueError("component panel and weights must share the same index")
    if not component_frame.columns.equals(weights.columns):
        raise ValueError("component panel and weights must share the same columns")
    if component_frame.shape[1] < 2:
        raise ValueError("need at least two constituents; correlation is undefined for one")

    _require_complete(weights, "weights")
    if not allow_nan:
        _require_complete(component_frame, "component panel")

    row_sums = weights.sum(axis=1)
    if (row_sums.sub(1.0).abs() > _WEIGHT_TOLERANCE).any():
        worst = row_sums.sub(1.0).abs().idxmax()
        raise ValueError(
            f"every row of weights must sum to 1; row {worst!r} sums to {row_sums[worst]:.10g}. "
            "Renormalise the subset before calling - see correlation.py on proxy baskets"
        )

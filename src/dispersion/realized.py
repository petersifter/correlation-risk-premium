"""Volatility primitives shared by the realised and implied measurements.

Small by design. This module owns exactly two pieces of arithmetic that more than one caller needs,
so that there is one place where each is defined and no chance of two copies drifting apart.

Why returns are not demeaned
----------------------------
:func:`annualized_vol` defaults to ``demean=False``, computing ``sqrt(mean(r^2) * 252)`` rather than
the sample standard deviation. Two reasons, and the second is the important one:

1. Over a 21-day window the mean daily return is almost entirely noise. Subtracting an estimate of
   something that small adds variance to the estimator rather than removing bias.
2. Implied volatility is a pure second-moment quantity - an option price says nothing about drift.
   Rung 4 compares realised against implied, and that comparison is only meaningful if both sides
   measure the same thing. Demeaning one side and not the other is a silent mismatch.

This is the standard convention for realised volatility in the options literature, and it is what
variance swaps actually pay on.

It also buys a convenient property that the test suite leans on hard. The matrix of non-demeaned
sample second moments *is* a covariance matrix, so when an index is reconstructed from its own
constituents, rung 1's identity holds **exactly** on sample moments - not approximately, not up to
sampling error. That turns the correctness check on the whole pipeline into a zero-tolerance
assertion rather than one that has to allow for noise.

Missing data
------------
Both functions reject ``NaN``. Real CRSP panels are full of holes - names listing, delisting,
halting - and deciding what to do about each hole is a research decision with consequences for
survivorship bias. That decision belongs where the holes are visible, not buried in an arithmetic
helper that would quietly drop observations. It matters concretely here: ``mean()`` skips ``NaN``
silently, so an unchecked hole would annualise over fewer observations than its neighbours and bias
a correlation with nothing raised.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "annualized_vol",
    "basket_return",
]

TRADING_DAYS_PER_YEAR = 252


def _require_complete(frame: pd.DataFrame | pd.Series, name: str) -> None:
    """Reject NaN. See the module docstring on why missingness is the caller's problem."""
    if frame.isna().to_numpy().any():
        raise ValueError(
            f"{name} contains NaN; resolve missing observations where the survivorship "
            "consequences of dropping or filling a name are visible, not here"
        )


def annualized_vol(
    returns: pd.DataFrame | pd.Series,
    *,
    demean: bool = False,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Any:
    """Annualised realised volatility over the whole of ``returns``.

    A DataFrame returns one volatility per column as a Series; a Series returns a scalar float.
    The return is annotated ``Any`` rather than a union because pandas ships no type stubs in this
    project, so mypy sees every pandas type as ``Any`` and overloads cannot discriminate on the
    argument - a precise union would be a fiction that only creates friction at call sites.

    With ``demean=False`` (the default, and the options-market convention)::

        sigma = sqrt( mean(r^2) * trading_days )

    With ``demean=True`` the sample standard deviation is used instead. See the module docstring on
    why the default does not demean.

    """
    _require_complete(returns, "returns")

    variance = returns.var(ddof=1) if demean else (returns**2).mean()
    result = np.sqrt(variance * trading_days)
    return float(result) if isinstance(returns, pd.Series) else result


def basket_return(component_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Return of the weighted basket, ``R_B = sum_i w_i R_i``.

    ``weights`` is indexed by the columns of ``component_returns`` and must sum to 1. The caller
    renormalises any subset before calling, deliberately, so that the choice stays visible at the
    call site - see ``correlation.py`` on proxy baskets.

    The basket built here is the *constructed* index, against which rung 1's identity is exact. It
    is the correctness check, not the measurement: the measurement uses the real index series, and
    the gap between the two is reported as ``basis``.
    """
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a Series indexed by the component columns")
    if not component_returns.columns.equals(weights.index):
        raise ValueError("component_returns columns and weights index must match exactly")
    if component_returns.shape[1] < 2:
        raise ValueError("need at least two constituents; correlation is undefined for one")

    _require_complete(component_returns, "component_returns")
    _require_complete(weights, "weights")

    if abs(float(weights.sum()) - 1.0) > 1e-8:
        raise ValueError(
            f"weights must sum to 1, got {float(weights.sum()):.10g}; renormalise before calling"
        )

    return (component_returns * weights).sum(axis=1)

"""Realized volatility and basket returns.

Volatility is not demeaned: sigma = sqrt(mean(r^2) * 252). Over 21 days the
sample mean is mostly noise, and implied volatility has no drift in it either,
so this keeps the two sides comparable. It is also what variance swaps settle
on. A side effect used in the tests: the matrix of raw second moments is itself
a valid covariance matrix, so a basket built from its own constituents
satisfies equation (1) in correlation.py exactly.

Both functions reject NaN. Whether to drop or fill a missing name is decided by
the caller, where the count can be recorded.
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
    if frame.isna().to_numpy().any():
        raise ValueError(f"{name} contains NaN; drop or fill missing names before calling")


def annualized_vol(
    returns: pd.DataFrame | pd.Series,
    *,
    demean: bool = False,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Any:
    """sqrt(mean(r^2) * trading_days), or the sample standard deviation if `demean`.

    One value per column for a DataFrame, a float for a Series. Typed as Any
    because pandas has no stubs installed here, so a union would only get in
    the way.
    """
    _require_complete(returns, "returns")

    variance = returns.var(ddof=1) if demean else (returns**2).mean()
    result = np.sqrt(variance * trading_days)
    return float(result) if isinstance(returns, pd.Series) else result


def basket_return(component_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """R_B = sum_i w_i R_i, with weights indexed by the return columns."""
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
            f"weights must sum to 1, got {float(weights.sum()):.10g}; renormalize before calling"
        )

    return (component_returns * weights).sum(axis=1)

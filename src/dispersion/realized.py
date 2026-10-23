from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _require_complete(frame: pd.DataFrame | pd.Series, name: str) -> None:
    if frame.isna().to_numpy().any():
        raise ValueError(f"{name} contains NaN")


def annualized_vol(
    returns: pd.DataFrame | pd.Series,
    *,
    demean: bool = False,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Any:
    """Not demeaned by default, matching implied vol and variance-swap conventions."""
    _require_complete(returns, "returns")

    variance = returns.var(ddof=1) if demean else (returns**2).mean()
    result = np.sqrt(variance * trading_days)
    return float(result) if isinstance(returns, pd.Series) else result


def basket_return(component_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a Series indexed by the component columns")
    if not component_returns.columns.equals(weights.index):
        raise ValueError("component_returns columns and weights index must match exactly")
    if component_returns.shape[1] < 2:
        raise ValueError("need at least two constituents")

    _require_complete(component_returns, "component_returns")
    _require_complete(weights, "weights")

    if abs(float(weights.sum()) - 1.0) > 1e-8:
        raise ValueError(f"weights must sum to 1, got {float(weights.sum()):.10g}")

    return (component_returns * weights).sum(axis=1)

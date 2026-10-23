"""The dispersion position, its P&L and its trading cost.

Short one index straddle with vega V_I and long a straddle on each basket name
with vega V_i = V_I w_i, so the vegas net to zero. Everything is per unit of
index vega, in volatility points.

For an at-the-money straddle held to expiry, delta-hedged P&L is approximately
vega * (sigma_realized - sigma_implied). Summed over both legs:

    P&L = sum_i w_i (sigma_r,i - sigma_imp,i) - (sigma_r,I - sigma_imp,I)
        = (A_r - sigma_r,I) - (A_imp - sigma_imp,I)

the realized volatility spread minus the implied one, with A the weighted
average constituent volatility. Cost is the quoted half-spread in volatility
points on each leg:

    cost = half_spread_index + sum_i w_i half_spread_i

A straddle has twice the dollar spread and twice the vega of one option, so its
half-spread in volatility points is the same as a single option's. Cost is
charged once because the 30-day options are held to expiry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Performance",
    "expanding_quantile_signal",
    "gross_pnl",
    "net_pnl",
    "performance",
    "trading_cost",
]

MONTHS_PER_YEAR = 12


def _require_columns(history: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in history.columns]
    if missing:
        raise ValueError(
            f"history is missing {missing}; run realized_correlation_history with implied=True"
        )


def gross_pnl(history: pd.DataFrame) -> pd.Series:
    """(A_r - sigma_r,I) - (A_imp - sigma_imp,I), in volatility points, before costs."""
    _require_columns(
        history,
        ["avg_single_vol", "index_vol", "avg_single_implied_vol", "index_implied_vol"],
    )
    realized_spread = history["avg_single_vol"] - history["index_vol"]
    implied_spread = history["avg_single_implied_vol"] - history["index_implied_vol"]
    return ((realized_spread - implied_spread) * 100.0).rename("gross_pnl_volpts")


def trading_cost(history: pd.DataFrame, *, cost_fraction: float = 1.0) -> pd.Series:
    """Entry cost in volatility points per unit of index vega.

    `cost_fraction` is the share of the quoted half-spread paid: 1.0 crosses it
    in full, 0.5 assumes half is saved by working orders.
    """
    _require_columns(history, ["index_half_spread_volpts", "basket_half_spread_volpts"])
    if not 0.0 <= cost_fraction <= 1.0:
        raise ValueError(f"cost_fraction must be in [0, 1], got {cost_fraction}")

    total = history["index_half_spread_volpts"] + history["basket_half_spread_volpts"]
    return (total * cost_fraction).rename("cost_volpts")


def net_pnl(
    history: pd.DataFrame,
    *,
    cost_fraction: float = 1.0,
    cost_multiple: float = 1.0,
) -> pd.Series:
    """Gross P&L minus cost. `cost_multiple=2` charges a round trip."""
    if cost_multiple <= 0.0:
        raise ValueError(f"cost_multiple must be positive, got {cost_multiple}")
    cost = trading_cost(history, cost_fraction=cost_fraction) * cost_multiple
    return (gross_pnl(history) - cost).rename("net_pnl_volpts")


@dataclass(frozen=True)
class Performance:
    """Summary of a monthly P&L series in volatility points per unit of index vega."""

    n_months: int
    mean: float
    volatility: float
    sharpe: float
    hit_rate: float
    worst_month: float
    max_drawdown: float
    total: float

    @property
    def sharpe_std_error(self) -> float:
        """sqrt((1 + SR^2 / 2) / T), T in years (Lo, 2002)."""
        return float(np.sqrt((1.0 + 0.5 * self.sharpe**2) / (self.n_months / MONTHS_PER_YEAR)))


def performance(pnl: pd.Series) -> Performance:
    """Summary statistics for a monthly P&L series.

    Sharpe is mean / std * sqrt(12) on P&L per unit of vega. A vega-neutral
    options book has no obvious capital base, so this is not a return on
    capital. Drawdown is on cumulative P&L in the same units.
    """
    values = pnl.dropna()
    if len(values) < 2:
        raise ValueError(f"need at least two months, got {len(values)}")

    cumulative = values.cumsum()
    drawdown = cumulative - cumulative.cummax()
    volatility = float(values.std(ddof=1))

    return Performance(
        n_months=len(values),
        mean=float(values.mean()),
        volatility=volatility,
        sharpe=float(values.mean() / volatility * np.sqrt(MONTHS_PER_YEAR)),
        hit_rate=float((values > 0).mean()),
        worst_month=float(values.min()),
        max_drawdown=float(drawdown.min()),
        total=float(cumulative.iloc[-1]),
    )


def expanding_quantile_signal(
    series: pd.Series,
    quantile: float = 0.75,
    min_periods: int = 36,
) -> pd.Series:
    """True where `series` is at or above the given quantile of its own past.

    The threshold is `.expanding().quantile().shift(1)`, so it only uses
    observations before the current one. The first `min_periods` values are
    False. A quantile over the whole sample would use future data.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be at least 2, got {min_periods}")

    threshold = series.expanding(min_periods=min_periods).quantile(quantile).shift(1)
    return (series >= threshold).fillna(value=False).rename("signal")

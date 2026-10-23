from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def _require_columns(history: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing {missing} (build it with implied=True)")


def gross_pnl(history: pd.DataFrame) -> pd.Series:
    _require_columns(
        history,
        ["avg_single_vol", "index_vol", "avg_single_implied_vol", "index_implied_vol"],
    )
    realized_spread = history["avg_single_vol"] - history["index_vol"]
    implied_spread = history["avg_single_implied_vol"] - history["index_implied_vol"]
    return ((realized_spread - implied_spread) * 100.0).rename("gross_pnl_volpts")


def trading_cost(history: pd.DataFrame, *, cost_fraction: float = 1.0) -> pd.Series:
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
    if cost_multiple <= 0.0:
        raise ValueError(f"cost_multiple must be positive, got {cost_multiple}")
    cost = trading_cost(history, cost_fraction=cost_fraction) * cost_multiple
    return (gross_pnl(history) - cost).rename("net_pnl_volpts")


@dataclass(frozen=True)
class Performance:
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
        return float(np.sqrt((1.0 + 0.5 * self.sharpe**2) / (self.n_months / MONTHS_PER_YEAR)))


def performance(pnl: pd.Series) -> Performance:
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
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be at least 2, got {min_periods}")

    # shift(1): the threshold only uses earlier observations
    threshold = series.expanding(min_periods=min_periods).quantile(quantile).shift(1)
    return (series >= threshold).fillna(value=False).rename("signal")

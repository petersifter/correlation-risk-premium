"""Rung 5: the dispersion position, its P&L, and what it costs to trade.

The position
------------
Short one index straddle, long a straddle on each of the 50 basket names, sized so the vegas net
to zero: if the index leg carries vega ``V_I``, name *i* carries ``V_i = V_I * w_i``. Everything
below is quoted **per unit of index vega, in volatility points**, which is the natural unit - it
makes the P&L independent of how large the book is and directly comparable with a bid-ask spread
measured the same way.

Why the P&L is a volatility spread
----------------------------------
A delta-hedged option earns ``integral of 0.5 * Gamma * S^2 * (sigma_r^2 - sigma_i^2) dt``. For an
at-the-money straddle held to expiry this collapses. Using the at-the-money approximations
``nu = 0.3989 * S * sqrt(T)`` and ``Gamma = 0.3989 / (S * sigma * sqrt(T))``::

    0.5 * Gamma * S^2 * T   ~  0.2 * S * sqrt(T) / sigma
    sigma_r^2 - sigma_i^2   =  (sigma_r - sigma_i)(sigma_r + sigma_i)  ~  2 sigma (sigma_r - sigma_i)

Multiplying, the ``sigma`` cancels and what is left is::

    P&L  ~  vega * (sigma_realised - sigma_implied)

Applying that to both legs and using ``V_i = V_I w_i``, the P&L per unit of index vega is::

    P&L = sum_i w_i (sigma_r,i - sigma_i,i) - (sigma_r,I - sigma_i,I)
        = (A_r - sigma_r,I) - (A_i - sigma_i,I)
        = realised volatility spread - implied volatility spread

where ``A = sum_i w_i sigma_i`` is the weighted-average constituent volatility from rung 1.

This is **not** the correlation premium
---------------------------------------
It is tempting to say the trade harvests the premium rung 4 measured. Measured over the sample, the
two series correlate at only +0.66 and disagree in sign in 20% of windows, so they are not the same
bet. The decomposition says why: of the +1.385 volatility points of average gross P&L, the short
index leg contributes +1.374 and the long single-name leg +0.010. Single-name options are priced
close to fair on average; index options are rich. **Vega-weighted dispersion is overwhelmingly a
short index volatility-risk-premium trade**, with the single-name leg acting as a hedge that is
roughly free but absorbs the volatility-level risk.

That distinction is the point of this rung. A study that assumed the P&L was the correlation
premium would attribute the result to the wrong exposure and size the risk against the wrong thing.

Costs
-----
Charged from quoted bid-ask, converted to volatility points by the contracts' own vega - see
:func:`dispersion.data.fetch_atm_quote_spreads`. Both legs pay::

    cost = half_spread_index + sum_i w_i half_spread_i

A straddle's half-spread in volatility points is the *same* as a single option's, which is
easy to get wrong in either direction. A straddle is a call plus a put, so its dollar spread is
roughly twice one option's - but so is its vega, and the two factors of two cancel::

    straddle spread in vol pts = (2 * dollar spread) / (2 * vega) = single-option spread in vol pts

So charging the single-contract half-spread per leg is right, and doubling it for "two options"
would double-count.

Two modelling choices, both stated rather than buried:

* **One-way, not round trip.** The options are 30 calendar days to expiry and the window is 21
  trading days, which is about 30 calendar days, so the position is held to expiry and pays entry
  only. Round-tripping would double the cost; since the base case already loses, the conclusion is
  robust to this choice and :func:`net_pnl` exposes ``cost_multiple`` to check it.
* **Crossing the spread.** Paying the full half-spread assumes the trade lifts the offer and hits
  the bid. A desk working orders would pay less, so ``cost_fraction`` scales it; the break-even
  fraction is the honest way to report how much execution skill the strategy would require.
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
    """Gross dispersion P&L in volatility points per unit of index vega, before costs.

    ``(A_realised - sigma_realised,I) - (A_implied - sigma_implied,I)``, the realised volatility
    spread minus the implied one. See the module docstring for the derivation.
    """
    _require_columns(
        history,
        ["avg_single_vol", "index_vol", "avg_single_implied_vol", "index_implied_vol"],
    )
    realized_spread = history["avg_single_vol"] - history["index_vol"]
    implied_spread = history["avg_single_implied_vol"] - history["index_implied_vol"]
    return ((realized_spread - implied_spread) * 100.0).rename("gross_pnl_volpts")


def trading_cost(history: pd.DataFrame, *, cost_fraction: float = 1.0) -> pd.Series:
    """Entry cost in volatility points per unit of index vega.

    ``cost_fraction`` scales the quoted half-spread: 1.0 crosses it in full, 0.5 assumes half of it
    is captured by working the order. It is a lever for asking how much execution skill the
    strategy needs, not a free parameter to tune until the result improves.
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
    """Gross P&L less trading cost, in volatility points per unit of index vega.

    ``cost_multiple`` of 2.0 charges a round trip instead of entry only - see the module docstring
    on why entry only is the base case.
    """
    if cost_multiple <= 0.0:
        raise ValueError(f"cost_multiple must be positive, got {cost_multiple}")
    cost = trading_cost(history, cost_fraction=cost_fraction) * cost_multiple
    return (gross_pnl(history) - cost).rename("net_pnl_volpts")


@dataclass(frozen=True)
class Performance:
    """Summary of a monthly P&L series, in volatility points per unit of index vega."""

    n_months: int
    mean: float
    volatility: float
    sharpe: float
    hit_rate: float
    worst_month: float
    max_drawdown: float
    total: float


def performance(pnl: pd.Series) -> Performance:
    """Summary statistics for a monthly P&L series.

    The Sharpe ratio here is on **P&L per unit of index vega**, annualised as
    ``mean / std * sqrt(12)``. It is not a return on capital: a vega-neutral options spread has no
    natural capital base without choosing a margin convention, and inventing one would make the
    number look precise while depending entirely on that choice. Quoted this way it compares
    like-for-like across cost assumptions, which is what this rung needs it for.

    Drawdown is on the cumulative undiscounted P&L, in the same units.
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
    """True where ``series`` is at or above the given quantile of everything observed *before* it.

    Rung 4 found that implied correlation over-reacts, which suggests trading only when it is high.
    Selecting those months with ``series.quantile(0.75)`` over the whole sample is lookahead: in
    1996 nobody knew the 1996-2024 quartile. The effect is not academic. On this data the
    full-sample threshold produces a mean net P&L of +0.245 volatility points and a Sharpe of 0.28,
    while the expanding threshold below produces -0.266 and -0.38. The entire apparent benefit of
    conditioning was the leak.

    So the threshold expands: at each date it is the quantile of the history up to and including the
    previous observation, via ``.expanding().quantile().shift(1)``. The shift matters - without it
    the current observation helps set the threshold it is then tested against. The first
    ``min_periods`` observations produce ``False`` rather than a threshold fitted on a handful of
    points.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if min_periods < 2:
        raise ValueError(f"min_periods must be at least 2, got {min_periods}")

    threshold = series.expanding(min_periods=min_periods).quantile(quantile).shift(1)
    return (series >= threshold).fillna(value=False).rename("signal")

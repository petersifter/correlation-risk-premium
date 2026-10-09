"""Rung 6: the tail, and what the strategy is actually exposed to.

Rung 5 found that vega-weighted dispersion earns almost all of its gross P&L from the index leg's
volatility risk premium. That invites the obvious challenge, and it is the one an interviewer will
make: **is this just short volatility wearing a costume?** This rung answers it with three views
rather than an assertion.

Why volatility alone cannot answer it
-------------------------------------
A short-volatility exposure and a short-crash exposure look identical in a volatility-only view,
because crashes are volatile. Separating them needs the market's *direction*, not just how far it
moved, which is why :mod:`dispersion.history` records ``index_return`` over each window alongside
``index_vol``.

The three views
---------------
:func:`volatility_exposure` regresses the P&L on the index volatility risk premium alone. A high
R-squared means the strategy is, statistically, that exposure and little else.

:func:`crash_exposure` regresses it on the market return **and the squared market return**. The
squared term is the informative one: a negative coefficient means the strategy loses when the
market moves a long way in *either* direction, which is the signature of a short-gamma position.
A negative linear term on top of that means the losses are worse on the downside specifically.
Reporting only the linear term would miss a position that is symmetric in direction but still
ruinous in size.

:func:`stress_buckets` sorts windows by market return and reports P&L by quintile. It carries no
functional-form assumption at all, which makes it the honest cross-check on the two regressions:
if the quintile pattern and the quadratic disagree, trust the quintiles.

A note on what "short vol" would mean here
------------------------------------------
Confirming the exposure is not a criticism of the measurement - it is the finding. A strategy that
is economically short index volatility should be described that way, sized against that risk, and
compared against simply selling index straddles, which is far cheaper to trade than 51 legs. The
value of this rung is that it makes the comparison unavoidable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dispersion.inference import HacRegression, hac_regression, newey_west_mean

__all__ = [
    "crash_exposure",
    "event_table",
    "index_volatility_premium",
    "stress_buckets",
    "volatility_exposure",
]


def index_volatility_premium(history: pd.DataFrame) -> pd.Series:
    """The index leg's volatility risk premium, in volatility points.

    ``sigma_implied,I - sigma_realised,I``. Positive when index options were rich, which is the
    quantity a short index volatility position is paid for.
    """
    for column in ("index_implied_vol", "index_vol"):
        if column not in history.columns:
            raise ValueError(f"history is missing {column!r}; run with implied=True")
    return ((history["index_implied_vol"] - history["index_vol"]) * 100.0).rename("index_vrp")


def volatility_exposure(pnl: pd.Series, history: pd.DataFrame) -> HacRegression:
    """Regress P&L on the index volatility risk premium alone.

    ``pnl_t = a + b * index_vrp_t + e_t``. The R-squared is the headline: it says how much of the
    strategy is explained by being short index volatility and nothing else.
    """
    return hac_regression(pnl, {"index_vrp": index_volatility_premium(history)})


def crash_exposure(pnl: pd.Series, history: pd.DataFrame) -> HacRegression:
    """Regress P&L on the market return and its square.

    ``pnl_t = a + b1 * r_t + b2 * r_t^2 + e_t`` with ``r`` the index return over the same window.

    ``b2 < 0`` is short gamma: large moves hurt regardless of direction. ``b1 > 0`` on top means the
    pain is concentrated on the downside. Both are needed - a position can be direction-neutral and
    still lose badly whenever the market moves, and a linear-only regression would call that
    "no market exposure".

    Returns are left in decimals, so ``b2`` is per unit of squared return; a 10% move contributes
    ``b2 * 0.01``.
    """
    if "index_return" not in history.columns:
        raise ValueError("history is missing 'index_return'; rebuild it with a current driver")

    market = history["index_return"]
    return hac_regression(pnl, {"market_return": market, "market_return_sq": market**2})


def stress_buckets(pnl: pd.Series, history: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    """Mean P&L by quintile of market return, with Newey-West intervals.

    Assumption-free: no functional form is imposed, so this is the check on whether the quadratic
    in :func:`crash_exposure` is telling the truth. The lowest bucket is the one that matters - it
    is where a short-crash exposure shows up as a large negative mean.
    """
    if "index_return" not in history.columns:
        raise ValueError("history is missing 'index_return'; rebuild it with a current driver")

    frame = pd.concat({"pnl": pnl, "market": history["index_return"]}, axis=1).dropna()
    if len(frame) < buckets * 3:
        raise ValueError(f"need at least {buckets * 3} observations for {buckets} buckets")

    labels = [f"Q{i + 1}" for i in range(buckets)]
    frame["bucket"] = pd.qcut(frame["market"], buckets, labels=labels)

    rows = []
    for label, chunk in frame.groupby("bucket", observed=True, sort=True):
        estimate = newey_west_mean(chunk["pnl"])
        rows.append(
            {
                "bucket": label,
                "n": estimate.n_obs,
                "market_return": float(chunk["market"].mean()),
                "mean_pnl": estimate.mean,
                "std_error": estimate.std_error,
                "ci_low": estimate.mean - 1.96 * estimate.std_error,
                "ci_high": estimate.mean + 1.96 * estimate.std_error,
                "worst": float(chunk["pnl"].min()),
            }
        )
    return pd.DataFrame(rows).set_index("bucket")


def event_table(
    pnl: pd.Series,
    history: pd.DataFrame,
    events: dict[str, str],
) -> pd.DataFrame:
    """The named episodes, in the measured quantities.

    ``events`` maps a label to the formation date whose forward window contains the episode. The
    window is what the position was exposed to, so the date given is the month the trade was *put
    on*, not the month the market moved - an easy off-by-one that would attribute each loss to the
    following month.
    """
    columns = [
        "implied_correlation",
        "realized_correlation",
        "index_implied_vol",
        "index_vol",
        "index_return",
    ]
    missing = [c for c in columns if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing {missing}")

    rows = []
    for label, date in events.items():
        key = pd.Timestamp(date)
        if key not in history.index:
            raise ValueError(f"{label}: {date} is not a formation date in this history")
        window = history.loc[key]
        rows.append(
            {
                "event": label,
                "formed": key.date(),
                "implied_corr": window["implied_correlation"],
                "realized_corr": window["realized_correlation"],
                "corr_surprise": window["realized_correlation"] - window["implied_correlation"],
                "index_iv": window["index_implied_vol"],
                "index_rv": window["index_vol"],
                "market_return": window["index_return"],
                "pnl": float(pnl.get(key, np.nan)),
            }
        )
    return pd.DataFrame(rows).set_index("event")

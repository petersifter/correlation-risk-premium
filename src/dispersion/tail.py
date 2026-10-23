from __future__ import annotations

import numpy as np
import pandas as pd

from dispersion.inference import HacRegression, hac_regression, newey_west_mean


def index_volatility_premium(history: pd.DataFrame) -> pd.Series:
    for column in ("index_implied_vol", "index_vol"):
        if column not in history.columns:
            raise ValueError(f"history is missing {column!r} (build it with implied=True)")
    return ((history["index_implied_vol"] - history["index_vol"]) * 100.0).rename("index_vrp")


def volatility_exposure(pnl: pd.Series, history: pd.DataFrame) -> HacRegression:
    return hac_regression(pnl, {"index_vrp": index_volatility_premium(history)})


def crash_exposure(pnl: pd.Series, history: pd.DataFrame) -> HacRegression:
    if "index_return" not in history.columns:
        raise ValueError("history is missing 'index_return'")

    market = history["index_return"]
    return hac_regression(pnl, {"market_return": market, "market_return_sq": market**2})


def stress_buckets(pnl: pd.Series, history: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    if "index_return" not in history.columns:
        raise ValueError("history is missing 'index_return'")

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

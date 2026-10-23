from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.inference import hac_regression
from dispersion.tail import (
    crash_exposure,
    event_table,
    index_volatility_premium,
    stress_buckets,
    volatility_exposure,
)


def synthetic_history(n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.008, 0.045, n)
    index_iv = np.abs(rng.normal(0.18, 0.05, n)) + 0.05
    index_rv = np.clip(index_iv - rng.normal(0.014, 0.03, n), 0.02, None)
    return pd.DataFrame(
        {
            "index_return": market,
            "index_implied_vol": index_iv,
            "index_vol": index_rv,
            "implied_correlation": rng.uniform(0.2, 0.7, n),
            "realized_correlation": rng.uniform(0.1, 0.8, n),
        },
        index=pd.bdate_range("2000-01-31", periods=n, freq="ME"),
    )


def test_vrp_units():
    history = pd.DataFrame({"index_implied_vol": [0.20], "index_vol": [0.15]})
    assert float(index_volatility_premium(history).iloc[0]) == pytest.approx(5.0)


def test_vrp_needs_implied():
    with pytest.raises(ValueError, match="implied=True"):
        index_volatility_premium(pd.DataFrame({"index_vol": [0.1]}))


def test_pure_short_vol_recovered():
    history = synthetic_history()
    pnl = 0.8 * index_volatility_premium(history)

    fit = volatility_exposure(pnl, history)
    assert fit.coefficients["index_vrp"] == pytest.approx(0.8)
    assert fit.r_squared == pytest.approx(1.0)


def test_short_gamma_recovered():
    history = synthetic_history()
    market = history["index_return"]
    pnl = 1.0 - 60.0 * market**2

    fit = crash_exposure(pnl, history)
    assert fit.coefficients["market_return_sq"] == pytest.approx(-60.0)
    assert fit.coefficients["market_return"] == pytest.approx(0.0, abs=1e-8)


def test_linear_only_fit_misreads_short_gamma():
    history = synthetic_history()
    market = history["index_return"]
    pnl = 1.0 - 60.0 * market**2

    linear_only = hac_regression(pnl, {"market_return": market})
    quadratic = crash_exposure(pnl, history)

    assert abs(linear_only.t_statistics["market_return"]) > 2.0
    assert quadratic.r_squared > 0.95
    assert linear_only.r_squared < quadratic.r_squared / 5.0
    assert quadratic.coefficients["market_return"] == pytest.approx(0.0, abs=1e-8)


def test_downside_term():
    history = synthetic_history()
    market = history["index_return"]
    pnl = 1.0 + 20.0 * market - 60.0 * market**2

    fit = crash_exposure(pnl, history)
    assert fit.coefficients["market_return"] == pytest.approx(20.0)
    assert fit.coefficients["market_return_sq"] == pytest.approx(-60.0)


def test_buckets_short_gamma():
    history = synthetic_history()
    pnl = 1.0 - 60.0 * history["index_return"] ** 2

    table = stress_buckets(pnl, history)
    assert list(table.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert table["market_return"].is_monotonic_increasing
    assert table.loc["Q3", "mean_pnl"] > table.loc["Q1", "mean_pnl"]
    assert table.loc["Q3", "mean_pnl"] > table.loc["Q5", "mean_pnl"]


def test_buckets_need_observations():
    history = synthetic_history(n=10)
    with pytest.raises(ValueError, match="at least 15 observations"):
        stress_buckets(pd.Series(1.0, index=history.index), history)


def test_crash_needs_market_return():
    history = synthetic_history().drop(columns=["index_return"])
    with pytest.raises(ValueError, match="index_return"):
        crash_exposure(pd.Series(dtype=float), history)


def test_event_table():
    history = synthetic_history()
    pnl = pd.Series(1.0, index=history.index)
    date = history.index[5]

    table = event_table(pnl, history, {"a quiet month": str(date.date())})
    assert table.loc["a quiet month", "formed"] == date.date()
    assert table.loc["a quiet month", "corr_surprise"] == pytest.approx(
        history.loc[date, "realized_correlation"] - history.loc[date, "implied_correlation"]
    )


def test_event_not_a_formation_date():
    history = synthetic_history()
    with pytest.raises(ValueError, match="not a formation date"):
        event_table(pd.Series(dtype=float), history, {"nope": "1066-10-14"})

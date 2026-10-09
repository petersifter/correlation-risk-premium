"""Tests for rung 6's exposure analysis.

The important property is that the three views agree when the truth is known. Each test builds a
history whose exposure is constructed, then checks the tool recovers it - a short-gamma position
must show a negative squared-return coefficient, a direction-neutral one must show no linear term,
and the assumption-free quintiles must agree with the quadratic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_index_volatility_premium_is_in_volatility_points():
    history = pd.DataFrame({"index_implied_vol": [0.20], "index_vol": [0.15]})
    assert float(index_volatility_premium(history).iloc[0]) == pytest.approx(5.0)


def test_premium_requires_the_implied_columns():
    with pytest.raises(ValueError, match="implied=True"):
        index_volatility_premium(pd.DataFrame({"index_vol": [0.1]}))


def test_a_pure_short_volatility_payoff_is_recovered():
    """P&L built as exactly 0.8x the index VRP must come back with b=0.8 and R-squared 1."""
    history = synthetic_history()
    pnl = 0.8 * index_volatility_premium(history)

    fit = volatility_exposure(pnl, history)
    assert fit.coefficients["index_vrp"] == pytest.approx(0.8)
    assert fit.r_squared == pytest.approx(1.0)


def test_short_gamma_shows_a_negative_squared_return_coefficient():
    """A payoff that loses on large moves in either direction is the short-gamma signature."""
    history = synthetic_history()
    market = history["index_return"]
    pnl = 1.0 - 60.0 * market**2  # symmetric in direction, punished by size

    fit = crash_exposure(pnl, history)
    assert fit.coefficients["market_return_sq"] == pytest.approx(-60.0)
    assert fit.coefficients["market_return"] == pytest.approx(0.0, abs=1e-8)


def test_a_linear_only_view_misreads_a_short_gamma_position():
    """Why the squared term is not optional, and why the linear one alone actively misleads.

    A payoff that is exactly symmetric in direction still produces a *significant* linear
    coefficient here, because the market return has a positive mean and so ``r`` and ``r^2`` are
    correlated in any real sample. Read on its own, that coefficient says "this position is short
    the market", which is false - the position does not care about direction at all. What gives the
    game away is explanatory power: the linear fit accounts for almost none of the variation, while
    adding the squared term accounts for nearly all of it.
    """
    from dispersion.inference import hac_regression

    history = synthetic_history()
    market = history["index_return"]
    pnl = 1.0 - 60.0 * market**2

    linear_only = hac_regression(pnl, {"market_return": market})
    quadratic = crash_exposure(pnl, history)

    assert abs(linear_only.t_statistics["market_return"]) > 2.0  # looks like market exposure
    assert quadratic.r_squared > 0.95  # the real structure
    # The linear view captures under a tenth of what is there, which is the tell.
    assert linear_only.r_squared < quadratic.r_squared / 5.0
    assert quadratic.coefficients["market_return"] == pytest.approx(0.0, abs=1e-8)


def test_downside_concentration_shows_in_the_linear_term():
    history = synthetic_history()
    market = history["index_return"]
    pnl = 1.0 + 20.0 * market - 60.0 * market**2

    fit = crash_exposure(pnl, history)
    assert fit.coefficients["market_return"] == pytest.approx(20.0)
    assert fit.coefficients["market_return_sq"] == pytest.approx(-60.0)


def test_stress_buckets_agree_with_the_quadratic():
    """Assumption-free check: a short-gamma payoff must be worst in the extreme quintiles."""
    history = synthetic_history()
    pnl = 1.0 - 60.0 * history["index_return"] ** 2

    table = stress_buckets(pnl, history)
    assert list(table.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert table["market_return"].is_monotonic_increasing
    assert table.loc["Q3", "mean_pnl"] > table.loc["Q1", "mean_pnl"]
    assert table.loc["Q3", "mean_pnl"] > table.loc["Q5", "mean_pnl"]


def test_stress_buckets_need_enough_observations():
    history = synthetic_history(n=10)
    with pytest.raises(ValueError, match="at least 15 observations"):
        stress_buckets(pd.Series(1.0, index=history.index), history)


def test_crash_exposure_requires_the_market_return():
    history = synthetic_history().drop(columns=["index_return"])
    with pytest.raises(ValueError, match="index_return"):
        crash_exposure(pd.Series(dtype=float), history)


def test_event_table_reports_the_formation_window():
    history = synthetic_history()
    pnl = pd.Series(1.0, index=history.index)
    date = history.index[5]

    table = event_table(pnl, history, {"a quiet month": str(date.date())})
    assert table.loc["a quiet month", "formed"] == date.date()
    assert table.loc["a quiet month", "corr_surprise"] == pytest.approx(
        history.loc[date, "realized_correlation"] - history.loc[date, "implied_correlation"]
    )


def test_event_table_rejects_a_date_that_is_not_a_formation_date():
    history = synthetic_history()
    with pytest.raises(ValueError, match="not a formation date"):
        event_table(pd.Series(dtype=float), history, {"nope": "1066-10-14"})

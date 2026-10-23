from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.strategy import (
    expanding_quantile_signal,
    gross_pnl,
    net_pnl,
    performance,
    trading_cost,
)


@pytest.fixture
def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "avg_single_implied_vol": [0.200, 0.300, 0.250],
            "index_implied_vol": [0.120, 0.200, 0.150],
            "avg_single_vol": [0.180, 0.320, 0.250],
            "index_vol": [0.100, 0.230, 0.150],
            "index_half_spread_volpts": [0.10, 0.40, 0.20],
            "basket_half_spread_volpts": [0.90, 3.60, 1.80],
        },
        index=pd.to_datetime(["2019-01-31", "2020-02-28", "2021-03-31"]),
    )


def test_gross_pnl(history):
    # Spreads (single - index), implied vs realized: 8 vs 8, 10 vs 9, 10 vs 10.
    pnl = gross_pnl(history)
    assert pnl.to_numpy() == pytest.approx([0.0, -1.0, 0.0])


def test_gross_pnl_equals_sum_of_legs(history):
    long_leg = (history["avg_single_vol"] - history["avg_single_implied_vol"]) * 100.0
    short_leg = -(history["index_vol"] - history["index_implied_vol"]) * 100.0

    pd.testing.assert_series_equal(
        gross_pnl(history), (long_leg + short_leg).rename("gross_pnl_volpts")
    )


def test_correlation_drop_pays():
    # Single-name vol as implied, index vol 4 points below implied.
    history = pd.DataFrame(
        {
            "avg_single_implied_vol": [0.30],
            "index_implied_vol": [0.20],
            "avg_single_vol": [0.30],
            "index_vol": [0.16],
            "index_half_spread_volpts": [0.0],
            "basket_half_spread_volpts": [0.0],
        },
        index=pd.to_datetime(["2020-01-31"]),
    )
    assert float(gross_pnl(history).iloc[0]) == pytest.approx(4.0)


def test_loses_when_all_vol_falls_even_if_correlation_falls():
    # Index/single ratio falls from 0.667 to 0.600, but the long leg loses 10 points
    # and the short leg only makes 8.
    history = pd.DataFrame(
        {
            "avg_single_implied_vol": [0.30],
            "index_implied_vol": [0.20],
            "avg_single_vol": [0.20],
            "index_vol": [0.12],
            "index_half_spread_volpts": [0.0],
            "basket_half_spread_volpts": [0.0],
        },
        index=pd.to_datetime(["2020-01-31"]),
    )
    pnl = float(gross_pnl(history).iloc[0])
    assert pnl == pytest.approx(-2.0)
    assert history["index_vol"].iloc[0] / history["avg_single_vol"].iloc[0] < (
        history["index_implied_vol"].iloc[0] / history["avg_single_implied_vol"].iloc[0]
    )


def test_cost_sums_legs(history):
    cost = trading_cost(history)
    assert cost.to_numpy() == pytest.approx([1.0, 4.0, 2.0])


def test_cost_fraction(history):
    half = trading_cost(history, cost_fraction=0.5)
    assert half.to_numpy() == pytest.approx([0.5, 2.0, 1.0])


def test_cost_fraction_range(history):
    with pytest.raises(ValueError, match="cost_fraction"):
        trading_cost(history, cost_fraction=1.5)


def test_net_pnl(history):
    net = net_pnl(history)
    assert net.to_numpy() == pytest.approx([-1.0, -5.0, -2.0])


def test_round_trip_cost(history):
    round_trip = net_pnl(history, cost_multiple=2.0)
    assert round_trip.to_numpy() == pytest.approx([-2.0, -9.0, -4.0])


def test_missing_implied_columns():
    with pytest.raises(ValueError, match="implied=True"):
        gross_pnl(pd.DataFrame({"avg_single_vol": [0.2], "index_vol": [0.1]}))


def test_performance():
    pnl = pd.Series([1.0, -2.0, 3.0, -1.0, 2.0])
    stats = performance(pnl)

    assert stats.n_months == 5
    assert stats.mean == pytest.approx(0.6)
    assert stats.total == pytest.approx(3.0)
    assert stats.hit_rate == pytest.approx(0.6)
    assert stats.worst_month == pytest.approx(-2.0)
    assert stats.sharpe == pytest.approx(0.6 / pnl.std(ddof=1) * np.sqrt(12))


def test_sharpe_std_error():
    stats = performance(pd.Series([1.0, -1.0] * 60 + [0.5]))
    years = stats.n_months / 12
    assert stats.sharpe_std_error == pytest.approx(np.sqrt((1 + 0.5 * stats.sharpe**2) / years))


def test_max_drawdown():
    # Cumulative 5, 3, 1, 4.
    pnl = pd.Series([5.0, -2.0, -2.0, 3.0])
    assert performance(pnl).max_drawdown == pytest.approx(-4.0)


def test_performance_needs_two_months():
    with pytest.raises(ValueError, match="at least two months"):
        performance(pd.Series([1.0]))


def test_expanding_signal_no_lookahead():
    # Truncating the series after t must not change the signal up to t.
    rng = np.random.default_rng(7)
    series = pd.Series(rng.uniform(0.1, 0.9, 200))

    full = expanding_quantile_signal(series, min_periods=24)
    for cut in (60, 120, 180):
        truncated = expanding_quantile_signal(series.iloc[:cut], min_periods=24)
        pd.testing.assert_series_equal(full.iloc[:cut], truncated)


def test_full_sample_quantile_fails_truncation():
    # A quiet first stretch then a high one: the full-sample bar is above everything early.
    rng = np.random.default_rng(7)
    quiet = rng.uniform(0.10, 0.30, 60)
    loud = rng.uniform(0.60, 0.90, 140)
    series = pd.Series(np.concatenate([quiet, loud]))

    leaky_full = series >= series.quantile(0.75)
    leaky_truncated = series.iloc[:60] >= series.iloc[:60].quantile(0.75)

    assert not leaky_full.iloc[:60].equals(leaky_truncated)
    assert not leaky_full.iloc[:60].any()
    assert leaky_truncated.any()

    expanding = expanding_quantile_signal(series, min_periods=24)
    pd.testing.assert_series_equal(
        expanding.iloc[:60], expanding_quantile_signal(series.iloc[:60], min_periods=24)
    )


def test_warmup_is_false():
    series = pd.Series(np.linspace(0.1, 0.9, 100))
    signal = expanding_quantile_signal(series, min_periods=36)
    assert not signal.iloc[:36].any()


def test_spike_fires():
    # Only possible if the current value is excluded from its own threshold.
    series = pd.Series([0.2] * 50 + [0.95] + [0.2] * 10)
    signal = expanding_quantile_signal(series, min_periods=24)
    assert bool(signal.iloc[50])


def test_invalid_quantile():
    with pytest.raises(ValueError, match="quantile must be"):
        expanding_quantile_signal(pd.Series([0.1, 0.2, 0.3]), quantile=1.0)

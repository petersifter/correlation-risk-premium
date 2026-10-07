"""Tests for rung 5's P&L accounting.

The load-bearing one is ``test_gross_pnl_decomposes_into_the_two_legs``: the P&L identity is
derived by cancelling terms, so the test asserts that the compact form really does equal the sum of
what each leg earns separately. If that ever fails, the derivation in the module docstring is wrong
and every number downstream is wrong with it.
"""

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
    """Three windows with hand-checkable volatilities, in decimals as the pipeline produces them."""
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


def test_gross_pnl_is_the_change_in_the_volatility_spread(history):
    """Window 1: implied spread 8.0 pts, realised spread 8.0 pts, so P&L is exactly zero.

    Window 2: implied 10.0, realised 9.0, so -1.0. Window 3: both 10.0, so zero.
    """
    pnl = gross_pnl(history)
    assert pnl.to_numpy() == pytest.approx([0.0, -1.0, 0.0])


def test_gross_pnl_decomposes_into_the_two_legs(history):
    """The compact identity must equal long-leg plus short-leg P&L computed separately.

    long  = A_realised - A_implied            (we own single-name vol)
    short = -(sigma_r,I - sigma_i,I)          (we are short index vol)
    """
    long_leg = (history["avg_single_vol"] - history["avg_single_implied_vol"]) * 100.0
    short_leg = -(history["index_vol"] - history["index_implied_vol"]) * 100.0

    pd.testing.assert_series_equal(
        gross_pnl(history), (long_leg + short_leg).rename("gross_pnl_volpts")
    )


def test_a_pure_correlation_fall_with_unchanged_vol_levels_is_profitable():
    """Hold single-name volatility fixed and let only the index leg come in lower.

    That is the clean case the trade is marketed on - correlation fell, nothing else moved - and it
    must pay. The realistic cases do not look like this, which is the point of the module docstring.
    """
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


def test_a_single_name_volatility_collapse_loses_despite_falling_correlation():
    """The trap. Realised correlation can fall while the trade still loses.

    Both legs' volatility comes in below implied, and the long single-name leg loses more than the
    short index leg gains. The index-to-single ratio falls from 0.667 to 0.600, so correlation fell
    - and the P&L is still negative.
    """
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


def test_trading_cost_sums_both_legs(history):
    cost = trading_cost(history)
    assert cost.to_numpy() == pytest.approx([1.0, 4.0, 2.0])


def test_cost_fraction_scales_the_half_spread(history):
    half = trading_cost(history, cost_fraction=0.5)
    assert half.to_numpy() == pytest.approx([0.5, 2.0, 1.0])


def test_cost_fraction_outside_the_unit_interval_is_rejected(history):
    with pytest.raises(ValueError, match="cost_fraction"):
        trading_cost(history, cost_fraction=1.5)


def test_net_pnl_subtracts_the_cost(history):
    net = net_pnl(history)
    assert net.to_numpy() == pytest.approx([-1.0, -5.0, -2.0])


def test_cost_multiple_charges_a_round_trip(history):
    round_trip = net_pnl(history, cost_multiple=2.0)
    assert round_trip.to_numpy() == pytest.approx([-2.0, -9.0, -4.0])


def test_missing_implied_columns_are_rejected():
    with pytest.raises(ValueError, match="implied=True"):
        gross_pnl(pd.DataFrame({"avg_single_vol": [0.2], "index_vol": [0.1]}))


def test_performance_statistics_on_a_known_series():
    pnl = pd.Series([1.0, -2.0, 3.0, -1.0, 2.0])
    stats = performance(pnl)

    assert stats.n_months == 5
    assert stats.mean == pytest.approx(0.6)
    assert stats.total == pytest.approx(3.0)
    assert stats.hit_rate == pytest.approx(0.6)
    assert stats.worst_month == pytest.approx(-2.0)
    assert stats.sharpe == pytest.approx(0.6 / pnl.std(ddof=1) * np.sqrt(12))


def test_max_drawdown_is_measured_on_cumulative_pnl():
    """Cumulative path 5, 3, 1, 4: the peak is 5 and the trough 1, so the drawdown is -4."""
    pnl = pd.Series([5.0, -2.0, -2.0, 3.0])
    assert performance(pnl).max_drawdown == pytest.approx(-4.0)


def test_performance_needs_at_least_two_months():
    with pytest.raises(ValueError, match="at least two months"):
        performance(pd.Series([1.0]))


def test_expanding_signal_cannot_see_the_future():
    """The no-lookahead guarantee, tested the way rung 2 tested its weights.

    Truncating the series after date t must not change the signal at or before t. A full-sample
    quantile fails this; an expanding one passes.
    """
    rng = np.random.default_rng(7)
    series = pd.Series(rng.uniform(0.1, 0.9, 200))

    full = expanding_quantile_signal(series, min_periods=24)
    for cut in (60, 120, 180):
        truncated = expanding_quantile_signal(series.iloc[:cut], min_periods=24)
        pd.testing.assert_series_equal(full.iloc[:cut], truncated)


def test_a_full_sample_quantile_would_fail_that_test():
    """Demonstrates the bug the expanding version exists to avoid, so the contrast is pinned down.

    A regime shift is what makes the leak bite: a quiet first half followed by a high second half
    pushes the full-sample quantile far above anything seen early, so months that looked extreme at
    the time are retrospectively judged ordinary. Implied correlation in this sample does exactly
    that, which is why the leak mattered here rather than being harmless.
    """
    rng = np.random.default_rng(7)
    quiet = rng.uniform(0.10, 0.30, 60)
    loud = rng.uniform(0.60, 0.90, 140)
    series = pd.Series(np.concatenate([quiet, loud]))

    leaky_full = series >= series.quantile(0.75)
    leaky_truncated = series.iloc[:60] >= series.iloc[:60].quantile(0.75)

    assert not leaky_full.iloc[:60].equals(leaky_truncated)
    assert not leaky_full.iloc[:60].any()  # nothing early clears the full-sample bar
    assert leaky_truncated.any()  # though a quarter of it cleared the bar at the time

    # The expanding version is unaffected by the later regime.
    honest = expanding_quantile_signal(series, min_periods=24)
    pd.testing.assert_series_equal(
        honest.iloc[:60], expanding_quantile_signal(series.iloc[:60], min_periods=24)
    )


def test_signal_excludes_the_warmup_period():
    series = pd.Series(np.linspace(0.1, 0.9, 100))
    signal = expanding_quantile_signal(series, min_periods=36)
    assert not signal.iloc[:36].any()


def test_the_current_observation_does_not_set_its_own_threshold():
    """A lone spike must fire, which it cannot if it is included in its own quantile."""
    series = pd.Series([0.2] * 50 + [0.95] + [0.2] * 10)
    signal = expanding_quantile_signal(series, min_periods=24)
    assert bool(signal.iloc[50])


def test_invalid_quantile_is_rejected():
    with pytest.raises(ValueError, match="quantile must be"):
        expanding_quantile_signal(pd.Series([0.1, 0.2, 0.3]), quantile=1.0)

"""Each figure renders from a synthetic panel, writes a file and closes itself."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dispersion.plots import (
    plot_correlation_premium,
    plot_premium_inference,
    plot_realized_correlation,
    plot_strategy_pnl,
    plot_tail_exposure,
)

ALL_PLOTS = [
    plot_realized_correlation,
    plot_correlation_premium,
    plot_premium_inference,
    plot_strategy_pnl,
    plot_tail_exposure,
]


@pytest.fixture
def panel() -> pd.DataFrame:
    n = 180
    rng = np.random.default_rng(11)
    index_iv = np.abs(rng.normal(0.18, 0.04, n)) + 0.06
    single_iv = index_iv + np.abs(rng.normal(0.10, 0.03, n))

    return pd.DataFrame(
        {
            "avg_single_vol": single_iv - rng.normal(0.01, 0.04, n),
            "index_vol": index_iv - rng.normal(0.014, 0.03, n),
            "avg_single_implied_vol": single_iv,
            "index_implied_vol": index_iv,
            "realized_correlation": rng.uniform(0.05, 0.9, n),
            "implied_correlation": rng.uniform(0.15, 0.8, n),
            "index_return": rng.normal(0.007, 0.042, n),
            "index_half_spread_volpts": np.abs(rng.normal(0.15, 0.08, n)),
            "basket_half_spread_volpts": np.abs(rng.normal(1.1, 0.5, n)),
        },
        index=pd.bdate_range("2004-01-30", periods=n, freq="ME"),
    )


@pytest.mark.parametrize("plot", ALL_PLOTS, ids=lambda f: f.__name__)
def test_renders(plot, panel, tmp_path):
    written = plot(panel, tmp_path / f"{plot.__name__}.png")

    assert written.exists()
    assert written.stat().st_size > 10_000


@pytest.mark.parametrize("plot", ALL_PLOTS, ids=lambda f: f.__name__)
def test_closes_figure(plot, panel, tmp_path):
    plt.close("all")
    plot(panel, tmp_path / f"{plot.__name__}.png")
    assert plt.get_fignums() == []


def test_creates_directory(panel, tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "chart.png"
    assert plot_realized_correlation(panel, nested).exists()


def test_missing_column(panel, tmp_path):
    with pytest.raises((KeyError, ValueError)):
        plot_strategy_pnl(panel.drop(columns=["basket_half_spread_volpts"]), tmp_path / "x.png")

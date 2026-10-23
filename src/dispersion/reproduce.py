"""Rebuild the panel, print the README's results and redraw its five figures.

    uv run dispersion-reproduce --username YOU   # rebuild from WRDS
    uv run dispersion-reproduce --from-cache     # reuse data/processed/

Sections are printed in README order. The cached panel is derived from licensed
data and is gitignored.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import pandas as pd

from dispersion.data import connect
from dispersion.history import realized_correlation_history
from dispersion.inference import (
    block_bootstrap_mean,
    forecast_regression,
    hac_regression,
    newey_west_mean,
    subsample_means,
)
from dispersion.plots import (
    plot_correlation_premium,
    plot_premium_inference,
    plot_realized_correlation,
    plot_strategy_pnl,
    plot_tail_exposure,
)
from dispersion.strategy import (
    expanding_quantile_signal,
    gross_pnl,
    net_pnl,
    performance,
    trading_cost,
)
from dispersion.tail import crash_exposure, event_table, stress_buckets, volatility_exposure

PANEL = Path("data/processed/correlation_history_top50.parquet")
REPORTS = Path("reports")
START, END = "1990-01-01", "2025-12-31"
TOP_N = 50

EVENTS = {
    "LTCM and Russia": "1998-07-31",
    "Lehman": "2008-09-30",
    "Euro sovereign crisis": "2011-07-29",
    "China devaluation": "2015-07-31",
    "Volmageddon": "2018-01-31",
    "Covid": "2020-02-28",
}


def build_panel(username: str) -> pd.DataFrame:
    """Rebuild the monthly panel from WRDS and cache it.

    CRSP starts the panel in 1990 and OptionMetrics in 1996, so earlier rows
    have no implied columns.
    """
    db = connect(username)
    try:
        panel = realized_correlation_history(
            db, START, END, top_n=TOP_N, implied=True, verbose=True
        )
    finally:
        db.close()

    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL)
    return panel


def _heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def report(panel: pd.DataFrame) -> None:
    with_implied = panel.dropna(subset=["implied_correlation"])
    tradeable = with_implied.dropna(subset=["basket_half_spread_volpts"])

    _heading("Sample")
    print(f"windows {len(panel)}, {panel.index[0]:%b %Y} to {panel.index[-1]:%b %Y}")
    print(f"skipped month-ends {len(panel.attrs.get('skipped_windows', []))}")
    print(f"with an implied leg {len(with_implied)}, with measured costs {len(tradeable)}")
    lost = panel["incomplete_in_window"]
    print(
        f"windows losing a constituent mid-window {int((lost > 0).sum())}, most lost {lost.max()}"
    )

    _heading("Realized correlation")
    rho = panel["realized_correlation"]
    print(f"mean {rho.mean():.3f}")
    decade = panel.groupby(panel.index.year // 10 * 10)
    print(
        pd.DataFrame(
            {
                "windows": decade.size(),
                "mean": decade["realized_correlation"].mean(),
                "max": decade["realized_correlation"].max(),
                "single_vol": decade["avg_single_vol"].mean(),
                "index_vol": decade["index_vol"].mean(),
            }
        )
        .round(3)
        .to_string()
    )
    print("highest windows:")
    print(rho.nlargest(4).round(3).to_string())

    _heading("The premium")
    premium = with_implied["implied_correlation"] - with_implied["realized_correlation"]
    print(
        f"windows {len(with_implied)}, {with_implied.index[0]:%b %Y} to "
        f"{with_implied.index[-1]:%b %Y}"
    )
    print(
        f"mean implied {with_implied['implied_correlation'].mean():.4f}, "
        f"mean realized {with_implied['realized_correlation'].mean():.4f}"
    )
    estimate = newey_west_mean(premium)
    print(
        f"mean {estimate.mean:.4f}, NW se {estimate.std_error:.4f}, naive se "
        f"{estimate.naive_std_error:.4f}, ratio {estimate.inflation:.2f} at {estimate.lags} lags"
    )
    print(
        f"t {estimate.t_statistic:.2f} (naive {estimate.mean / estimate.naive_std_error:.2f}), "
        f"positive in {(premium > 0).mean():.1%}"
    )

    fit = forecast_regression(
        with_implied["realized_correlation"], with_implied["implied_correlation"]
    )
    print(f"alpha {fit.alpha:.4f} (se {fit.alpha_std_error:.4f}, t vs 0 {fit.t_alpha:.2f})")
    print(f"beta {fit.beta:.4f} (se {fit.beta_std_error:.4f}, t vs 1 {fit.t_beta_equals_one:.2f})")
    print(f"R-squared {fit.r_squared:.3f}, n {fit.n_obs}, {fit.lags} lags")
    print(f"fitted line meets the 45-degree line at {fit.alpha / (1 - fit.beta):.2f}")
    decades = pd.Series(with_implied.index.year // 10 * 10, index=with_implied.index)
    print(subsample_means(premium, decades).round(4).to_string())

    _heading("The position and its cost")
    gross, cost, net = gross_pnl(tradeable), trading_cost(tradeable), net_pnl(tradeable)
    for label, series in [("gross", gross), ("cost", cost), ("net", net)]:
        m = newey_west_mean(series)
        print(f"{label:>5} mean {m.mean:.3f}, NW se {m.std_error:.3f}, t {m.t_statistic:.2f}")
    print(f"cost / gross {cost.mean() / gross.mean():.1%}")

    stats = performance(net)
    low, high = block_bootstrap_mean(net)
    print(
        f"Sharpe {stats.sharpe:.2f} +/- {stats.sharpe_std_error:.2f}, hit rate {stats.hit_rate:.1%}"
    )
    print(
        f"worst month {stats.worst_month:.2f} ({stats.worst_month / stats.volatility:.1f} sd), "
        f"max drawdown {stats.max_drawdown:.1f}, cumulative {stats.total:.1f}"
    )
    print(f"skew {net.skew():.2f}, excess kurtosis {net.kurt():.2f}")
    print(f"block bootstrap 95% interval for the mean {low:.3f} to {high:.3f}")
    era = pd.DataFrame({"gross": gross, "cost": cost, "net": net}).groupby(
        tradeable.index.year // 5 * 5
    )
    print(era.mean().round(3).to_string())

    _heading("Conditioning on implied correlation")
    implied = tradeable["implied_correlation"]
    full_sample = implied >= implied.quantile(0.75)
    expanding = expanding_quantile_signal(implied)
    for label, signal in [("full-sample quartile", full_sample), ("expanding quartile", expanding)]:
        s = performance(net[signal])
        print(
            f"{label}: {s.n_months} windows, mean net {s.mean:.3f}, "
            f"Sharpe {s.sharpe:.2f} +/- {s.sharpe_std_error:.2f}"
        )

    _heading("Exposure")
    index_leg = (tradeable["index_implied_vol"] - tradeable["index_vol"]) * 100
    single_leg = (tradeable["avg_single_vol"] - tradeable["avg_single_implied_vol"]) * 100
    print(f"short index leg {index_leg.mean():.3f}, long single-name leg {single_leg.mean():.3f}")
    print(f"corr(gross, premium) {gross.corr(premium.reindex(gross.index)):.3f}")
    vol_fit = volatility_exposure(net, tradeable)
    print(
        f"on index VRP: slope {vol_fit.coefficients['index_vrp']:.3f} "
        f"(t {vol_fit.t_statistics['index_vrp']:.1f}), R-squared {vol_fit.r_squared:.3f}"
    )
    vrp_fit = hac_regression(-single_leg, {"index_vrp": index_leg})
    print(
        f"single-name VRP on index VRP: corr {index_leg.corr(-single_leg):.3f}, "
        f"slope {vrp_fit.coefficients['index_vrp']:.3f}, R-squared {vrp_fit.r_squared:.3f}"
    )
    crash = crash_exposure(net, tradeable)
    print(crash.summary().round(3).to_string())
    print(f"R-squared {crash.r_squared:.3f}")
    surprise = tradeable["realized_correlation"] - tradeable["implied_correlation"]
    corr_fit = hac_regression(net, {"surprise": surprise})
    print(
        f"on correlation surprise: slope {corr_fit.coefficients['surprise']:.2f} "
        f"(t {corr_fit.t_statistics['surprise']:.1f}), R-squared {corr_fit.r_squared:.3f}"
    )
    print(stress_buckets(net, tradeable).round(3).to_string())
    print(event_table(net, tradeable, EVENTS).round(3).to_string())

    _heading("Limitations")
    early = tradeable.index.year < 2010
    print(
        "median names without a two-sided quote: "
        f"{tradeable.loc[early, 'names_without_quotes'].median():.0f} before 2010, "
        f"{tradeable.loc[~early, 'names_without_quotes'].median():.0f} after"
    )
    tenor = tradeable["cost_tenor_days"].groupby(tradeable.index.year // 5 * 5).median()
    print("median cost tenor (days) by period:")
    print(tenor.to_string())


def draw(panel: pd.DataFrame) -> list[Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    return [
        plot_realized_correlation(panel, REPORTS / "realized_correlation.png"),
        plot_correlation_premium(panel, REPORTS / "correlation_premium.png"),
        plot_premium_inference(panel, REPORTS / "premium_inference.png"),
        plot_strategy_pnl(panel, REPORTS / "strategy_pnl.png"),
        plot_tail_exposure(panel, REPORTS / "tail_exposure.png"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the panel and print the results.")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=f"load {PANEL} instead of rebuilding from WRDS",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("WRDS_USERNAME"),
        help="WRDS username, or set WRDS_USERNAME (the password comes from .pgpass)",
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)

    if args.from_cache:
        if not PANEL.exists():
            parser.error(f"{PANEL} does not exist; run without --from-cache to build it")
        panel = pd.read_parquet(PANEL)
        print(f"loaded {len(panel)} windows from {PANEL}")
    else:
        if not args.username:
            parser.error("pass --username or set WRDS_USERNAME, or use --from-cache")
        print("rebuilding from WRDS (a few minutes, mostly option queries)")
        panel = build_panel(args.username)

    report(panel)

    _heading("Figures")
    for written in draw(panel):
        print(f"  {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

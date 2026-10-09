"""Rebuild the panel, print the README's statistics and redraw its five figures.

    uv run dispersion-reproduce --username YOU   # rebuild from WRDS
    uv run dispersion-reproduce --from-cache     # reuse data/processed/

The cached panel is derived from licensed data and is gitignored.
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
from dispersion.strategy import gross_pnl, net_pnl, performance, trading_cost
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

    _heading("Realized correlation")
    rho = panel["realized_correlation"]
    print(f"windows {len(panel)}, {panel.index[0]:%b %Y} to {panel.index[-1]:%b %Y}")
    print(f"mean {rho.mean():.3f}")
    decade = panel.groupby(panel.index.year // 10 * 10)
    print(decade["realized_correlation"].agg(["size", "mean", "max"]).round(3).to_string())
    print("highest windows:")
    print(rho.nlargest(4).round(3).to_string())

    _heading("The premium")
    premium = with_implied["implied_correlation"] - with_implied["realized_correlation"]
    print(f"windows with both legs {len(with_implied)}")
    print(
        f"mean implied {with_implied['implied_correlation'].mean():.4f}, "
        f"mean realized {with_implied['realized_correlation'].mean():.4f}"
    )
    estimate = newey_west_mean(premium)
    print(
        f"mean {estimate.mean:.4f}, NW se {estimate.std_error:.4f}, naive se "
        f"{estimate.naive_std_error:.4f}, ratio {estimate.inflation:.2f} at {estimate.lags} lags"
    )
    print(f"t {estimate.t_statistic:.2f}, positive in {(premium > 0).mean():.1%}")
    fit = forecast_regression(
        with_implied["realized_correlation"], with_implied["implied_correlation"]
    )
    print(f"alpha {fit.alpha:.4f} (se {fit.alpha_std_error:.4f}, t vs 0 {fit.t_alpha:.2f})")
    print(f"beta {fit.beta:.4f} (se {fit.beta_std_error:.4f}, t vs 1 {fit.t_beta_equals_one:.2f})")
    print(f"R-squared {fit.r_squared:.3f}")
    decades = pd.Series(with_implied.index.year // 10 * 10, index=with_implied.index)
    print(subsample_means(premium, decades).round(4).to_string())

    _heading("The position and its cost")
    gross, cost, net = gross_pnl(tradeable), trading_cost(tradeable), net_pnl(tradeable)
    print(f"windows {len(tradeable)}")
    for label, series in [("gross", gross), ("cost", cost), ("net", net)]:
        m = newey_west_mean(series)
        print(f"{label:>5} mean {m.mean:.3f}, NW se {m.std_error:.3f}, t {m.t_statistic:.2f}")
    stats = performance(net)
    print(
        f"Sharpe {stats.sharpe:.2f}, hit rate {stats.hit_rate:.1%}, worst {stats.worst_month:.2f}, "
        f"max drawdown {stats.max_drawdown:.1f}, cumulative {stats.total:.1f}"
    )
    era = pd.DataFrame({"gross": gross, "cost": cost, "net": net}).groupby(
        tradeable.index.year // 5 * 5
    )
    print(era.mean().round(3).to_string())

    _heading("Exposure")
    index_leg = (tradeable["index_implied_vol"] - tradeable["index_vol"]) * 100
    single_leg = (tradeable["avg_single_vol"] - tradeable["avg_single_implied_vol"]) * 100
    print(f"short index leg {index_leg.mean():.3f}, long single-name leg {single_leg.mean():.3f}")
    vol_fit = volatility_exposure(net, tradeable)
    print(
        f"on index VRP: slope {vol_fit.coefficients['index_vrp']:.3f} "
        f"(t {vol_fit.t_statistics['index_vrp']:.1f}), R-squared {vol_fit.r_squared:.3f}"
    )
    print(f"corr(index VRP, single-name VRP) {index_leg.corr(-single_leg):.3f}")
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

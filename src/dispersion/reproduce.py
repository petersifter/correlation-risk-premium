"""Regenerate every figure and every number quoted in the README.

The point of this module is that nothing in the README is unreproducible. Run it and you get the
four figures in ``reports/`` and a printed report containing each statistic the README cites, in
the order the README cites them.

Two modes::

    uv run dispersion-reproduce              # rebuild from WRDS, then draw
    uv run dispersion-reproduce --from-cache # redraw from the saved panel, no credentials needed

The cached panel is aggregate output rather than licensed record-level data, but it is still
derived from licensed sources, so it is gitignored like everything else under ``data/``. A reader
with WRDS credentials runs the first form; a reader checking the drawing and statistics code
against a panel they already built runs the second.

Building from WRDS takes a couple of minutes, almost all of it the per-formation-date option
queries. Nothing here is clever: it is the same functions the rest of the package exposes, called
in order, so that the published results and the tested code cannot drift apart.
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
    "Aug 1998 LTCM/Russia": "1998-07-31",
    "Lehman, Oct 2008": "2008-09-30",
    "Euro crisis, Aug 2011": "2011-07-29",
    "Aug 2015 devaluation": "2015-07-31",
    "Volmageddon, Feb 2018": "2018-01-31",
    "Covid, Mar 2020": "2020-02-28",
}


def build_panel(username: str) -> pd.DataFrame:
    """Rebuild the monthly panel from WRDS and cache it.

    Rung 2 runs from 1990 because CRSP does; the implied legs start in 1996 because OptionMetrics
    does, so the early rows carry realised measurements with no implied counterpart. That is why
    every downstream statistic drops rows missing the implied columns rather than assuming the
    panel is rectangular.
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
    """Print every statistic the README quotes, in the order it quotes them."""
    realized_only = panel
    with_implied = panel.dropna(subset=["implied_correlation"])
    tradeable = with_implied.dropna(subset=["basket_half_spread_volpts"])

    _heading("Rung 2 - realised correlation")
    rho = realized_only["realized_correlation"]
    print(
        f"windows {len(realized_only)}, {realized_only.index[0]:%b %Y} to "
        f"{realized_only.index[-1]:%b %Y}"
    )
    print(f"mean realised correlation {rho.mean():.3f}")
    decade = realized_only.assign(d=realized_only.index.year // 10 * 10).groupby("d")
    print(decade["realized_correlation"].agg(["size", "mean", "max"]).round(3).to_string())
    print("highest windows:")
    print(rho.nlargest(4).round(3).to_string())

    _heading("Rung 3 - implied correlation")
    premium = with_implied["implied_correlation"] - with_implied["realized_correlation"]
    print(f"windows with both legs {len(with_implied)}")
    print(
        f"mean implied {with_implied['implied_correlation'].mean():.4f}, "
        f"mean realised {with_implied['realized_correlation'].mean():.4f}"
    )
    print(f"mean premium {premium.mean():+.4f}, positive in {(premium > 0).mean():.1%}")

    _heading("Rung 4 - inference")
    estimate = newey_west_mean(premium)
    print(
        f"mean {estimate.mean:+.4f}  NW se {estimate.std_error:.4f} "
        f"(naive {estimate.naive_std_error:.4f}, inflation {estimate.inflation:.2f}x, "
        f"{estimate.lags} lags)"
    )
    print(f"t {estimate.t_statistic:.2f}")
    fit = forecast_regression(
        with_implied["realized_correlation"], with_implied["implied_correlation"]
    )
    print(f"alpha {fit.alpha:+.4f} (se {fit.alpha_std_error:.4f}, t vs 0 {fit.t_alpha:+.2f})")
    print(
        f"beta  {fit.beta:+.4f} (se {fit.beta_std_error:.4f}, t vs 1 {fit.t_beta_equals_one:+.2f})"
    )
    print(f"R-squared {fit.r_squared:.3f}")
    print(
        subsample_means(
            premium, pd.Series(with_implied.index.year // 10 * 10, index=with_implied.index)
        )
        .round(4)
        .to_string()
    )

    _heading("Rung 5 - the position and its costs")
    gross, cost, net = gross_pnl(tradeable), trading_cost(tradeable), net_pnl(tradeable)
    index_leg = (tradeable["index_implied_vol"] - tradeable["index_vol"]) * 100
    single_leg = (tradeable["avg_single_vol"] - tradeable["avg_single_implied_vol"]) * 100
    print(f"windows {len(tradeable)}")
    for label, series in [("gross", gross), ("cost", cost), ("net", net)]:
        m = newey_west_mean(series)
        print(f"  {label:>6} mean {m.mean:+.3f}  NW se {m.std_error:.3f}  t {m.t_statistic:+.2f}")
    print(
        f"  leg decomposition: short index {index_leg.mean():+.3f}, "
        f"long single names {single_leg.mean():+.3f}"
    )
    print(
        f"  gross vs premium: corr {gross.corr(premium.reindex(gross.index)):+.3f}, "
        f"sign disagreement {((gross > 0) != (premium.reindex(gross.index) > 0)).mean():.1%}"
    )
    stats = performance(net)
    print(
        f"  Sharpe {stats.sharpe:+.2f}  hit rate {stats.hit_rate:.1%}  "
        f"worst {stats.worst_month:+.2f}  max drawdown {stats.max_drawdown:+.1f}  "
        f"total {stats.total:+.1f}"
    )
    era = (
        pd.DataFrame({"gross": gross, "cost": cost, "net": net})
        .groupby(tradeable.index.year // 5 * 5)
        .mean()
    )
    print(era.round(3).to_string())

    _heading("Rung 6 - what the exposure actually is")
    vol_fit = volatility_exposure(net, tradeable)
    print(
        f"on index VRP: beta {vol_fit.coefficients['index_vrp']:+.3f} "
        f"(t {vol_fit.t_statistics['index_vrp']:+.1f}), R-squared {vol_fit.r_squared:.3f}"
    )
    index_vrp = index_leg
    single_vrp = -single_leg
    print(f"corr(index VRP, single-name VRP) {index_vrp.corr(single_vrp):+.3f}")
    crash = crash_exposure(net, tradeable)
    print(crash.summary().round(3).to_string())
    print(f"R-squared {crash.r_squared:.3f}")
    surprise = tradeable["realized_correlation"] - tradeable["implied_correlation"]
    corr_fit = hac_regression(net, {"surprise": surprise})
    print(
        f"on correlation surprise: slope {corr_fit.coefficients['surprise']:+.2f} "
        f"(t {corr_fit.t_statistics['surprise']:+.1f}), R-squared {corr_fit.r_squared:.3f}"
    )
    print(stress_buckets(net, tradeable).round(3).to_string())
    print(event_table(net, tradeable, EVENTS).round(3).to_string())


def draw(panel: pd.DataFrame) -> list[Path]:
    """Write every figure the README embeds."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    return [
        plot_realized_correlation(panel, REPORTS / "realized_correlation.png"),
        plot_correlation_premium(panel, REPORTS / "correlation_premium.png"),
        plot_premium_inference(panel, REPORTS / "premium_inference.png"),
        plot_strategy_pnl(panel, REPORTS / "strategy_pnl.png"),
        plot_tail_exposure(panel, REPORTS / "tail_exposure.png"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=f"redraw from {PANEL} instead of rebuilding from WRDS",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("WRDS_USERNAME"),
        help=(
            "WRDS username, or set WRDS_USERNAME. The password is never passed here - it comes "
            "from your .pgpass file, created once with db.create_pgpass_file()."
        ),
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)

    if not args.from_cache and not args.username:
        parser.error(
            "a WRDS username is required to rebuild: pass --username or set WRDS_USERNAME "
            "(or use --from-cache to redraw from an existing panel)"
        )

    if args.from_cache:
        if not PANEL.exists():
            parser.error(f"{PANEL} does not exist; run without --from-cache to build it")
        panel = pd.read_parquet(PANEL)
        print(f"loaded {len(panel)} windows from {PANEL}")
    else:
        print("rebuilding from WRDS - a couple of minutes, mostly option queries")
        panel = build_panel(args.username)
        print(f"built {len(panel)} windows and cached them to {PANEL}")

    report(panel)

    _heading("Figures")
    for written in draw(panel):
        print(f"  {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

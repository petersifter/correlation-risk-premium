"""Charts for the dispersion project.

Colours are the validated default palette documented in the ``dataviz`` skill's
``references/palette.md``: categorical slot 1 blue ``#2a78d6`` and slot 2 orange ``#eb6834``, an
adjacent pair whose worst-case colour-vision-deficiency separation is Delta E 9.1 on the light
surface against a >= 8 target. They are used unchanged rather than re-picked by eye.

These figures commit to a single light look. They are PNGs destined for a README rather than a
themed web page, so there is no dark variant to select steps for.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

__all__ = [
    "plot_correlation_premium",
    "plot_premium_inference",
    "plot_realized_correlation",
    "plot_strategy_pnl",
    "plot_tail_exposure",
]


def _separated_peaks(series: pd.Series, count: int, min_years: int = 3) -> pd.Series:
    """The ``count`` highest observations, no two within ``min_years`` of each other.

    Taking a plain ``nlargest`` clusters every label on one episode - the four highest windows in
    this sample all fall between April 2010 and October 2011 - which renders as a pile of
    overlapping text that says one thing four times. Greedily taking the maximum and then excluding
    its neighbourhood yields labels for distinct episodes, which is what a reader wants from them.
    """
    remaining = series.dropna().sort_values(ascending=False)
    chosen: dict[pd.Timestamp, float] = {}

    for date, value in remaining.items():
        if len(chosen) == count:
            break
        if all(abs((date - picked).days) > min_years * 365 for picked in chosen):
            chosen[date] = value

    return pd.Series(chosen).sort_index()


def _recede(ax: plt.Axes) -> None:
    """Push axes furniture into the background so the data is the only prominent thing."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def plot_realized_correlation(
    history: pd.DataFrame,
    path: str | Path,
    *,
    annotate: int = 4,
) -> Path:
    """Two stacked panels sharing a time axis.

    Top panel: realised average correlation, one series, so no legend box - the title names it.
    The ``annotate`` highest windows carry direct labels; the rest do not, because a number on
    every point is noise.

    Bottom panel: weighted-average single-name volatility against the actual S&P 500 index
    volatility. Both are volatilities in the same unit, so they share one scale. Plotting
    correlation and volatility against two y-axes on one panel would be a dual-axis chart, which
    is the most common way to make a chart lie about relative magnitude - hence two panels.

    The bottom panel is the point of the figure: the *gap* between the two lines is the
    diversification the short index leg of a dispersion trade is sold against, and it closes
    exactly when correlation spikes.
    """
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        height_ratios=[3, 2],
        gridspec_kw={"hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)

    rho = history["realized_correlation"]
    mean_rho = float(rho.mean())

    top.plot(rho.index, rho, color=SERIES_1, linewidth=1.6, zorder=3)
    top.axhline(mean_rho, color=TEXT_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    top.annotate(
        f"mean {mean_rho:.2f}",
        xy=(rho.index[-1], mean_rho),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        color=TEXT_SECONDARY,
        fontsize=9,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
    )

    for date, value in _separated_peaks(rho, annotate).items():
        top.plot([date], [value], marker="o", markersize=5, color=SERIES_1, zorder=4)
        near_right = date > rho.index[int(len(rho) * 0.82)]
        top.annotate(
            f"{date:%b %Y}  {value:.2f}",
            xy=(date, value),
            xytext=(-7 if near_right else 7, 7),
            textcoords="offset points",
            ha="right" if near_right else "left",
            color=TEXT_PRIMARY,
            fontsize=9,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        )

    top.set_ylim(-0.1, 1.08)
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    top.set_ylabel("average correlation", color=TEXT_SECONDARY, fontsize=10)
    top.set_title(
        "Realised average correlation of the 50 largest S&P 500 constituents",
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=14,
    )
    _recede(top)

    bottom.plot(
        history.index,
        history["avg_single_vol"],
        color=SERIES_2,
        linewidth=1.6,
        label="weighted-average single-name volatility",
        zorder=3,
    )
    bottom.plot(
        history.index,
        history["index_vol"],
        color=SERIES_1,
        linewidth=1.6,
        label="S&P 500 index volatility",
        zorder=3,
    )
    bottom.fill_between(
        history.index,
        history["index_vol"],
        history["avg_single_vol"],
        color=SERIES_2,
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    bottom.set_ylim(0, None)
    bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    bottom.set_ylabel("annualised volatility", color=TEXT_SECONDARY, fontsize=10)
    bottom.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
    )
    _recede(bottom)

    fig.text(
        0.125,
        0.035,
        "Monthly baskets formed on point-in-time index membership and market-cap weights, held "
        f"21 trading days forward. {len(history)} windows, "
        f"{history.index[0]:%b %Y} to {history.index[-1]:%b %Y}. "
        "The shaded gap is the diversification benefit; it closes as correlation rises.",
        color=TEXT_SECONDARY,
        fontsize=8.5,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_premium(history: pd.DataFrame, path: str | Path) -> Path:
    """Implied against realised correlation, and the gap between them.

    Colour carries one meaning throughout: blue is what the options market *charged*, orange is
    what the stocks *delivered*. The lower panel inherits that - the premium is implied minus
    realised, so blue above zero means the charge exceeded the outcome and the seller of
    correlation won, orange below zero means the reverse. The same hue never changes meaning
    between panels.
    """
    both = history.dropna(subset=["implied_correlation"])
    premium = both["implied_correlation"] - both["realized_correlation"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 2], gridspec_kw={"hspace": 0.12}
    )
    fig.patch.set_facecolor(SURFACE)

    top.plot(
        both.index,
        both["implied_correlation"],
        color=SERIES_1,
        linewidth=1.5,
        label="implied correlation (what the market charged)",
        zorder=3,
    )
    top.plot(
        both.index,
        both["realized_correlation"],
        color=SERIES_2,
        linewidth=1.5,
        alpha=0.85,
        label="realised correlation (what the stocks delivered)",
        zorder=3,
    )
    top.set_ylim(-0.1, 1.02)
    top.set_ylabel("average correlation", color=TEXT_SECONDARY, fontsize=10)
    top.set_title(
        "The correlation risk premium: implied versus subsequently realised",
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=14,
    )
    top.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _recede(top)

    bottom.axhline(0.0, color=TEXT_SECONDARY, linewidth=1.0, zorder=2)
    bottom.fill_between(
        premium.index, premium, 0, where=premium >= 0, color=SERIES_1, alpha=0.55, linewidth=0
    )
    bottom.fill_between(
        premium.index, premium, 0, where=premium < 0, color=SERIES_2, alpha=0.65, linewidth=0
    )
    bottom.axhline(
        float(premium.mean()),
        color=TEXT_PRIMARY,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    bottom.annotate(
        f"mean +{premium.mean():.3f}",
        xy=(premium.index[-1], premium.mean()),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        color=TEXT_PRIMARY,
        fontsize=9,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
    )

    worst = premium.idxmin()
    bottom.annotate(
        f"{worst:%b %Y}  {premium.loc[worst]:.2f}",
        xy=(worst, premium.loc[worst]),
        xytext=(7, 2),
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=9,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
    )

    # A real MINUS SIGN, not a hyphen: this is rendered chart text, where it is the correct
    # glyph rather than an ASCII lookalike.
    bottom.set_ylabel("implied − realised", color=TEXT_SECONDARY, fontsize=10)  # noqa: RUF001
    _recede(bottom)

    fig.text(
        0.125,
        0.035,
        f"{len(both)} monthly windows, {both.index[0]:%b %Y} to {both.index[-1]:%b %Y}. "
        "Implied is the 30-day 50-delta surface on the formation date; realised is the following "
        "21 trading days. Both legs use the identical top-50 basket and weights, so the proxy "
        "approximation largely cancels between them.",
        color=TEXT_SECONDARY,
        fontsize=8.5,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_premium_inference(
    history: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Two panels: is the forecast unbiased, and has the premium decayed.

    Left, the Mincer-Zarnowitz scatter. The grey 45-degree line is what an unbiased forecast would
    trace; the orange line is the fitted relationship. A fitted slope visibly flatter than 45
    degrees *is* the over-reaction result - implied correlation moves around more than realised
    correlation does.

    Right, the decade means with robust 95% intervals. Point estimates alone invite the reader to
    see a trend whether or not one is there, so this panel exists to show whether the intervals
    actually separate - and whether the most recent one still clears zero.
    """
    from dispersion.inference import forecast_regression, subsample_means

    both = history.dropna(subset=["implied_correlation"])
    premium = both["implied_correlation"] - both["realized_correlation"]
    fit = forecast_regression(both["realized_correlation"], both["implied_correlation"])
    decades = subsample_means(premium, pd.Series(both.index.year // 10 * 10, index=both.index))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor(SURFACE)

    grid = np.linspace(0.0, 1.0, 50)
    left.plot(
        grid,
        grid,
        color=TEXT_SECONDARY,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        label="unbiased forecast (slope 1)",
        zorder=2,
    )
    left.scatter(
        both["implied_correlation"],
        both["realized_correlation"],
        s=14,
        color=SERIES_1,
        alpha=0.45,
        linewidth=0,
        zorder=3,
    )
    left.plot(
        grid,
        fit.alpha + fit.beta * grid,
        color=SERIES_2,
        linewidth=2.0,
        label=f"fitted (slope {fit.beta:.2f})",
        zorder=4,
    )

    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_xlabel("implied correlation", color=TEXT_SECONDARY, fontsize=10)
    left.set_ylabel("subsequently realised correlation", color=TEXT_SECONDARY, fontsize=10)
    left.set_title(
        "Implied correlation over-reacts", color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12
    )
    left.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _recede(left)

    positions = np.arange(len(decades))
    right.axvline(0.0, color=TEXT_SECONDARY, linewidth=1.0, zorder=2)
    for y, (_, row) in zip(positions, decades.iterrows(), strict=True):
        clears_zero = row["ci_low"] > 0
        colour = SERIES_1 if clears_zero else SERIES_2
        right.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2.0, zorder=3)
        right.plot([row["mean"]], [y], marker="o", markersize=7, color=colour, zorder=4)
        right.annotate(
            f"{row['mean']:+.3f}",
            xy=(row["ci_high"], y),
            xytext=(8, -3),
            textcoords="offset points",
            color=TEXT_PRIMARY,
            fontsize=9,
        )

    right.set_yticks(positions)
    right.set_yticklabels([f"{int(d)}s" for d in decades.index])
    right.invert_yaxis()
    right.set_xlim(-0.03, 0.25)
    # A real MINUS SIGN, not a hyphen: rendered chart text, where it is the correct glyph.
    premium_label = "premium (implied − realised), 95% interval"  # noqa: RUF001
    right.set_xlabel(premium_label, color=TEXT_SECONDARY, fontsize=10)
    right.set_title(
        "and the premium has decayed", color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12
    )
    _recede(right)

    fig.text(
        0.125,
        0.015,
        f"{len(both)} monthly windows, {both.index[0]:%b %Y} to {both.index[-1]:%b %Y}. "
        "Intervals are Newey-West with Bartlett weights at the automatic lag. The 2020s "
        "interval includes zero.",
        color=TEXT_SECONDARY,
        fontsize=8.5,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_strategy_pnl(history: pd.DataFrame, path: str | Path) -> Path:
    """Cumulative P&L before and after measured costs, and the era breakdown.

    Colour keeps its meaning from the other figures: blue is what the trade earns, orange is what
    it costs. The top panel's two lines diverging *is* the result - the gap between them is the
    bid-ask, and by the end it has consumed essentially all of the gross.
    """
    from dispersion.strategy import gross_pnl, net_pnl, trading_cost

    both = history.dropna(subset=["implied_correlation", "basket_half_spread_volpts"])
    gross, cost, net = gross_pnl(both), trading_cost(both), net_pnl(both)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7.4), height_ratios=[3, 2], gridspec_kw={"hspace": 0.32}
    )
    fig.patch.set_facecolor(SURFACE)

    top.plot(
        gross.index,
        gross.cumsum(),
        color=SERIES_1,
        linewidth=1.8,
        label="gross, before trading costs",
        zorder=3,
    )
    top.plot(
        net.index,
        net.cumsum(),
        color=SERIES_2,
        linewidth=1.8,
        label="net of measured bid-ask",
        zorder=4,
    )
    top.fill_between(
        gross.index, gross.cumsum(), net.cumsum(), color=SERIES_2, alpha=0.10, linewidth=0, zorder=1
    )
    top.axhline(0.0, color=TEXT_SECONDARY, linewidth=1.0, zorder=2)
    top.set_ylabel("cumulative volatility points", color=TEXT_SECONDARY, fontsize=10)
    top.set_title(
        "Transaction costs consume the dispersion premium",
        color=TEXT_PRIMARY,
        fontsize=13,
        loc="left",
        pad=14,
    )
    top.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _recede(top)

    era = (both.index.year // 5 * 5).astype(int)
    table = pd.DataFrame({"gross": gross, "cost": cost, "net": net}).groupby(era).mean()
    positions = np.arange(len(table))

    bottom.axhline(0.0, color=TEXT_SECONDARY, linewidth=1.0, zorder=3)
    for x, (_, row) in zip(positions, table.iterrows(), strict=True):
        colour = SERIES_1 if row["net"] > 0 else SERIES_2
        bottom.bar(x, row["net"], width=0.55, color=colour, zorder=2)
        # Value label just outside the bar end, so positive and negative bars never collide.
        bottom.annotate(
            f"{row['net']:+.2f}",
            xy=(x, row["net"]),
            xytext=(0, 5 if row["net"] > 0 else -5),
            textcoords="offset points",
            ha="center",
            va="bottom" if row["net"] > 0 else "top",
            color=TEXT_PRIMARY,
            fontsize=9,
        )
        # Components pinned to one fixed row below every bar, rather than offset from each one -
        # an offset from a negative bar lands on top of that bar's own label.
        bottom.annotate(
            f"gross {row['gross']:.2f}\ncost {row['cost']:.2f}",
            xy=(x, -2.0),
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=8,
        )

    bottom.set_xticks(positions)
    bottom.set_xticklabels([f"{y}-{str(y + 4)[-2:]}" for y in table.index])
    bottom.set_ylim(-2.4, 3.4)
    bottom.set_ylabel("mean net, vol points", color=TEXT_SECONDARY, fontsize=10)
    bottom.set_title(
        "Profitable only in the late 1990s", color=TEXT_PRIMARY, fontsize=12, loc="left", pad=10
    )
    _recede(bottom)

    fig.text(
        0.125,
        0.005,
        f"{len(both)} monthly windows, {both.index[0]:%b %Y} to {both.index[-1]:%b %Y}. "
        "Per unit of index vega. Costs are quoted OptionMetrics bid-ask converted to "
        "volatility points by contract vega, charged one way. Commissions and delta-hedging "
        "costs are not included, so the net shown is an upper bound.",
        color=TEXT_SECONDARY,
        fontsize=8.5,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tail_exposure(history: pd.DataFrame, path: str | Path) -> Path:
    """What the strategy is actually exposed to: market moves, or correlation?

    Two scatters on identical y-axes so the contrast is read directly off the page rather than off
    two R-squared values in prose. Left is net P&L against the market's return over the window;
    right is the same P&L against the correlation surprise. If the trade were short volatility or
    short crash, the left panel would show the structure. It does not. The right panel does.
    """
    from dispersion.inference import hac_regression
    from dispersion.strategy import net_pnl

    both = history.dropna(subset=["implied_correlation", "basket_half_spread_volpts"])
    pnl = net_pnl(both)
    market = both["index_return"]
    surprise = both["realized_correlation"] - both["implied_correlation"]

    quadratic = hac_regression(pnl, {"r": market, "r2": market**2})
    linear = hac_regression(pnl, {"surprise": surprise})

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5.2), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    fig.patch.set_facecolor(SURFACE)

    left.scatter(market, pnl, s=16, color=SERIES_1, alpha=0.45, linewidth=0, zorder=3)
    grid = np.linspace(market.min(), market.max(), 100)
    left.plot(
        grid,
        quadratic.coefficients["const"]
        + quadratic.coefficients["r"] * grid
        + quadratic.coefficients["r2"] * grid**2,
        color=TEXT_SECONDARY,
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        zorder=4,
        label=f"quadratic fit, R² = {quadratic.r_squared:.3f}",
    )
    left.set_xlabel("market return over the window", color=TEXT_SECONDARY, fontsize=10)
    left.set_ylabel("net P&L, volatility points", color=TEXT_SECONDARY, fontsize=10)
    left.set_title("Not a market bet", color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12)
    left.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    left.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    _recede(left)

    right.scatter(surprise, pnl, s=16, color=SERIES_2, alpha=0.45, linewidth=0, zorder=3)
    grid = np.linspace(surprise.min(), surprise.max(), 100)
    right.plot(
        grid,
        linear.coefficients["const"] + linear.coefficients["surprise"] * grid,
        color=TEXT_SECONDARY,
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        zorder=4,
        label=f"linear fit, R² = {linear.r_squared:.3f}",
    )
    # A real MINUS SIGN, not a hyphen: rendered chart text, where it is the correct glyph.
    surprise_label = "correlation surprise (realised − implied)"  # noqa: RUF001
    right.set_xlabel(surprise_label, color=TEXT_SECONDARY, fontsize=10)
    right.set_title("A correlation bet", color=TEXT_PRIMARY, fontsize=12, loc="left", pad=12)
    right.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    _recede(right)

    fig.text(
        0.125,
        0.015,
        f"{len(both)} monthly windows. The single-name leg hedges the volatility level - "
        "index and single-name volatility risk premia correlate at +0.91 - leaving "
        "correlation as the exposure that remains.",
        color=TEXT_SECONDARY,
        fontsize=8.5,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path

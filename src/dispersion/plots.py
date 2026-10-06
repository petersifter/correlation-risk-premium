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
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

__all__ = ["plot_realized_correlation"]


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

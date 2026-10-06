"""Figures for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

BLUE, ORANGE, GREY = "tab:blue", "tab:orange", "grey"

__all__ = [
    "plot_correlation_premium",
    "plot_realized_correlation",
]


def _tidy(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _separated_peaks(series: pd.Series, count: int, min_years: int = 3) -> pd.Series:
    """The `count` highest values, at least `min_years` apart.

    The four highest windows are all in 2010-2011, so plain nlargest would put
    four labels on top of each other.
    """
    remaining = series.dropna().sort_values(ascending=False)
    chosen: dict[pd.Timestamp, float] = {}

    for date, value in remaining.items():
        if len(chosen) == count:
            break
        if all(abs((date - picked).days) > min_years * 365 for picked in chosen):
            chosen[date] = value

    return pd.Series(chosen).sort_index()


def plot_realized_correlation(
    history: pd.DataFrame,
    path: str | Path,
    *,
    annotate: int = 4,
) -> Path:
    """Realized correlation (top) and single-name vs index volatility (bottom)."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[3, 2], gridspec_kw={"hspace": 0.1}
    )

    rho = history["realized_correlation"]
    top.plot(rho.index, rho, color=BLUE, lw=1.3)
    top.axhline(rho.mean(), color=GREY, lw=1, ls="--", label=f"mean {rho.mean():.2f}")
    for date, value in _separated_peaks(rho, annotate).items():
        top.annotate(
            f"{date:%b %Y}",
            xy=(date, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    top.set_ylim(-0.1, 1.05)
    top.set_ylabel("average correlation")
    top.set_title(
        f"Realized correlation of the 50 largest S&P 500 stocks, "
        f"{history.index[0]:%b %Y} to {history.index[-1]:%b %Y}"
    )
    top.legend(loc="upper left", frameon=False)
    _tidy(top)

    bottom.plot(history.index, history["avg_single_vol"], color=ORANGE, lw=1.3, label="single-name")
    bottom.plot(history.index, history["index_vol"], color=BLUE, lw=1.3, label="S&P 500")
    bottom.set_ylim(0, None)
    bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    bottom.set_ylabel("realized volatility")
    bottom.legend(loc="upper left", frameon=False)
    _tidy(bottom)

    return _save(fig, path)


def plot_correlation_premium(history: pd.DataFrame, path: str | Path) -> Path:
    """Implied and realized correlation (top) and their difference (bottom)."""
    both = history.dropna(subset=["implied_correlation"])
    premium = both["implied_correlation"] - both["realized_correlation"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[3, 2], gridspec_kw={"hspace": 0.1}
    )

    top.plot(both.index, both["implied_correlation"], color=BLUE, lw=1.3, label="implied")
    top.plot(both.index, both["realized_correlation"], color=ORANGE, lw=1.3, label="realized")
    top.set_ylim(-0.1, 1.05)
    top.set_ylabel("average correlation")
    top.set_title(
        "Implied correlation on the formation date and realized correlation over the next month"
    )
    top.legend(loc="upper left", frameon=False)
    _tidy(top)

    bottom.bar(premium.index, premium, width=25, color=np.where(premium >= 0, BLUE, ORANGE))
    bottom.axhline(0.0, color="black", lw=0.8)
    bottom.axhline(premium.mean(), color=GREY, lw=1, ls="--", label=f"mean {premium.mean():.3f}")
    bottom.set_ylabel("implied - realized")
    bottom.legend(loc="lower left", frameon=False)
    _tidy(bottom)

    return _save(fig, path)

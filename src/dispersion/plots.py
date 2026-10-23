from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from dispersion.inference import forecast_regression, hac_regression, subsample_means
from dispersion.strategy import gross_pnl, net_pnl, trading_cost

BLUE, ORANGE, GREY = "tab:blue", "tab:orange", "grey"


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


def plot_premium_inference(history: pd.DataFrame, path: str | Path) -> Path:
    both = history.dropna(subset=["implied_correlation"])
    premium = both["implied_correlation"] - both["realized_correlation"]
    fit = forecast_regression(both["realized_correlation"], both["implied_correlation"])
    decades = subsample_means(premium, pd.Series(both.index.year // 10 * 10, index=both.index))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.25})

    grid = np.linspace(0.0, 1.0, 50)
    left.scatter(
        both["implied_correlation"], both["realized_correlation"], s=12, color=BLUE, alpha=0.5
    )
    left.plot(grid, grid, color=GREY, lw=1, ls="--", label="45-degree line")
    left.plot(
        grid,
        fit.alpha + fit.beta * grid,
        color=ORANGE,
        lw=1.8,
        label=f"OLS fit, slope {fit.beta:.2f}",
    )
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_xlabel("implied correlation")
    left.set_ylabel("realized correlation, next 21 days")
    left.set_title("Realized on implied correlation")
    left.legend(loc="upper left", frameon=False)
    _tidy(left)

    positions = np.arange(len(decades))
    right.errorbar(
        decades["mean"],
        positions,
        xerr=[decades["mean"] - decades["ci_low"], decades["ci_high"] - decades["mean"]],
        fmt="o",
        color=BLUE,
        capsize=4,
    )
    right.axvline(0.0, color="black", lw=0.8)
    right.set_yticks(positions, [f"{int(d)}s" for d in decades.index])
    right.invert_yaxis()
    right.set_xlabel("mean premium (implied - realized), Newey-West 95% CI")
    right.set_title("Premium by decade")
    _tidy(right)

    return _save(fig, path)


def plot_strategy_pnl(history: pd.DataFrame, path: str | Path) -> Path:
    both = history.dropna(subset=["implied_correlation", "basket_half_spread_volpts"])
    gross, cost, net = gross_pnl(both), trading_cost(both), net_pnl(both)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.8), height_ratios=[3, 2], gridspec_kw={"hspace": 0.35}
    )

    top.plot(gross.index, gross.cumsum(), color=BLUE, lw=1.5, label="gross")
    top.plot(net.index, net.cumsum(), color=ORANGE, lw=1.5, label="net of quoted half-spread")
    top.axhline(0.0, color="black", lw=0.8)
    top.set_ylabel("cumulative vol points")
    top.set_title("Dispersion P&L per unit of index vega")
    top.legend(loc="upper left", frameon=False)
    _tidy(top)

    era = (both.index.year // 5 * 5).astype(int)
    table = pd.DataFrame({"gross": gross, "cost": cost, "net": net}).groupby(era).mean()
    x = np.arange(len(table))
    width = 0.27
    bottom.bar(x - width, table["gross"], width, color=BLUE, label="gross")
    bottom.bar(x, table["cost"], width, color=GREY, label="cost")
    bottom.bar(x + width, table["net"], width, color=ORANGE, label="net")
    bottom.axhline(0.0, color="black", lw=0.8)
    bottom.set_xticks(x, [f"{y}-{str(y + 4)[-2:]}" for y in table.index])
    bottom.set_ylabel("mean per month, vol points")
    bottom.set_title("By five-year period")
    bottom.legend(loc="upper right", frameon=False, ncols=3)
    _tidy(bottom)

    return _save(fig, path)


def plot_tail_exposure(history: pd.DataFrame, path: str | Path) -> Path:
    both = history.dropna(subset=["implied_correlation", "basket_half_spread_volpts"])
    pnl = net_pnl(both)
    market = both["index_return"]
    surprise = both["realized_correlation"] - both["implied_correlation"]

    quadratic = hac_regression(pnl, {"r": market, "r2": market**2})
    linear = hac_regression(pnl, {"surprise": surprise})

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"wspace": 0.08}
    )

    left.scatter(market, pnl, s=12, color=BLUE, alpha=0.5)
    grid = np.linspace(market.min(), market.max(), 100)
    left.plot(
        grid,
        quadratic.coefficients["const"]
        + quadratic.coefficients["r"] * grid
        + quadratic.coefficients["r2"] * grid**2,
        color="black",
        lw=1.2,
        ls="--",
        label=f"quadratic fit, R2 = {quadratic.r_squared:.3f}",
    )
    left.set_xlabel("S&P 500 return over the window")
    left.set_ylabel("net P&L, vol points")
    left.set_title("Net P&L vs market return")
    left.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    left.legend(loc="lower left", frameon=False)
    _tidy(left)

    right.scatter(surprise, pnl, s=12, color=ORANGE, alpha=0.5)
    grid = np.linspace(surprise.min(), surprise.max(), 100)
    right.plot(
        grid,
        linear.coefficients["const"] + linear.coefficients["surprise"] * grid,
        color="black",
        lw=1.2,
        ls="--",
        label=f"linear fit, R2 = {linear.r_squared:.3f}",
    )
    right.set_xlabel("realized - implied correlation")
    right.set_title("Net P&L vs correlation surprise")
    right.legend(loc="lower left", frameon=False)
    _tidy(right)

    return _save(fig, path)

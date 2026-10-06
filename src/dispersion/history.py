"""The monthly correlation panel.

On each formation date t (the last trading day of a month) the basket is fixed
from index membership and market-cap weights as of t, then held for the next
`window` trading days, over which volatilities are measured. Forming the basket
at the end of the window instead would use knowledge of which names survived
it; on 2008-2012 that raises mean realized correlation by 0.0038, in 53 of 53
windows.

Each window carries two index volatilities. `index_vol` is from CRSP's sprtrn,
the actual S&P 500, and is the measurement. `basket_vol` is from the basket's
own weighted return, against which equation (1) in correlation.py holds
exactly. `basis = index_vol - basket_vol` is the tracking error of a top-50,
market-cap-weighted proxy.

Two counts are kept per window: `dropped_at_formation` (members with no usable
market cap on t) and `incomplete_in_window` (weighted names whose forward
returns have gaps, usually because they were acquired or delisted mid-window).
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from dispersion.correlation import average_correlation
from dispersion.data import (
    SPX_SECID,
    InsufficientData,
    fetch_atm_implied_vols,
    fetch_atm_quote_spreads,
    fetch_index_membership,
    fetch_index_returns,
    fetch_optionmetrics_link,
    fetch_stock_panel,
    formation_weights,
    members_on,
)
from dispersion.realized import TRADING_DAYS_PER_YEAR, annualized_vol, basket_return

__all__ = [
    "correlation_window",
    "implied_correlation_on",
    "realized_correlation_history",
]


def correlation_window(
    window_returns: pd.DataFrame,
    index_window_returns: pd.Series,
    weights: pd.Series,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float | int]:
    """Measure one held window. No data access.

    `window_returns` has dates on the index and permnos as columns;
    `index_window_returns` covers the same dates; `weights` is the formation-date
    basket. Names with gaps in the window are dropped, the rest renormalized,
    and the number dropped is returned.
    """
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a Series indexed by permno")
    missing = weights.index.difference(window_returns.columns)
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} weighted name(s) are absent from window_returns entirely, e.g. "
            f"{list(missing[:3])}. Fetch the union of members across the whole period, not the "
            "members as of a single date"
        )
    if not index_window_returns.index.equals(window_returns.index):
        raise ValueError("index and constituent windows must cover exactly the same dates")
    # Constituents with gaps are filtered below; the index has no such filter, and mean()
    # would skip a NaN and annualize the index over fewer days than the constituents.
    if index_window_returns.isna().any():
        n_missing = int(index_window_returns.isna().sum())
        raise ValueError(
            f"index_window_returns has {n_missing} missing observation(s) out of "
            f"{len(index_window_returns)}; volatility would be annualized over a different number "
            "of days than the constituents"
        )

    candidates = window_returns[weights.index]
    usable = candidates.columns[candidates.notna().all()]
    incomplete = int(len(weights) - len(usable))

    if len(usable) < 2:
        raise InsufficientData(
            f"only {len(usable)} name(s) have complete returns over the window; correlation needs "
            "two"
        )

    renormalized = weights[usable] / weights[usable].sum()
    component_returns = candidates[usable]

    component_vols = annualized_vol(component_returns, trading_days=trading_days)
    weighted_avg_vol = float(renormalized @ component_vols)

    index_vol = float(annualized_vol(index_window_returns, trading_days=trading_days))
    basket_vol = float(
        annualized_vol(basket_return(component_returns, renormalized), trading_days=trading_days)
    )

    vols = component_vols.to_numpy()
    w = renormalized.to_numpy()

    return {
        "n_names": len(usable),
        "incomplete_in_window": incomplete,
        "avg_single_vol": weighted_avg_vol,
        "index_vol": index_vol,
        "basket_vol": basket_vol,
        "basis": index_vol - basket_vol,
        "realized_correlation": average_correlation(index_vol, vols, w),
        "basket_correlation": average_correlation(basket_vol, vols, w),
    }


def implied_correlation_on(
    db: Any,
    as_of: pd.Timestamp,
    weights: pd.Series,
    *,
    days: int = 30,
) -> dict[str, float | int]:
    """Implied correlation on `as_of` for the same basket as the realized leg.

    Same names and weights, only the volatilities differ, so the top-50 proxy
    error is common to both legs. Names with no OptionMetrics link or no surface
    on the date are dropped and the rest renormalized. Also returns quoted
    half-spreads for the basket and the index, and the median cost tenor.
    """
    as_of_str = as_of.strftime("%Y-%m-%d")
    link = fetch_optionmetrics_link(db, list(weights.index), as_of_str)
    if link.empty:
        raise InsufficientData(f"no OptionMetrics links for any basket name on {as_of.date()}")

    surface = fetch_atm_implied_vols(
        db, [*link.to_numpy().tolist(), SPX_SECID], as_of_str, days=days
    )
    if SPX_SECID not in surface.index:
        raise InsufficientData(f"no SPX surface on {as_of.date()}")

    index_iv = float(surface.loc[SPX_SECID, "atm_vol"])
    if not index_iv > 0:
        # SPX sometimes has a surface row with a NULL vol (late July / early August 2020).
        raise InsufficientData(
            f"SPX has a surface row on {as_of.date()} but no usable ATM volatility"
        )
    usable_links = link[link.isin(surface.index) & link.ne(SPX_SECID)]
    component_iv = surface.loc[usable_links.to_numpy(), "atm_vol"]
    component_iv = component_iv[component_iv.notna()]

    usable = usable_links[usable_links.isin(component_iv.index)]
    if len(usable) < 2:
        raise InsufficientData(
            f"only {len(usable)} basket name(s) have an ATM surface on {as_of.date()}"
        )

    w = weights.loc[usable.index]
    w = w / w.sum()
    vols = component_iv.loc[usable.to_numpy()].to_numpy()

    skew_gap = float(
        (surface.loc[usable.to_numpy(), "call_vol"] - surface.loc[usable.to_numpy(), "put_vol"])
        .abs()
        .mean()
    )

    # Names with no two-sided quote are left out of the weighted cost and counted.
    spread_table = fetch_atm_quote_spreads(db, [*usable.to_numpy().tolist(), SPX_SECID], as_of_str)
    spreads = (
        spread_table["half_spread_volpts"] if not spread_table.empty else pd.Series(dtype=float)
    )
    index_half_spread = float(spreads.get(SPX_SECID, float("nan")))

    priced = usable[usable.isin(spreads.index) & usable.ne(SPX_SECID)]
    if priced.empty:
        basket_half_spread = float("nan")
    else:
        cost_weights = weights.loc[priced.index]
        cost_weights = cost_weights / cost_weights.sum()
        basket_half_spread = float(
            cost_weights.to_numpy() @ spreads.loc[priced.to_numpy()].to_numpy()
        )

    return {
        "n_names_implied": len(usable),
        "unlinked_or_unpriced": int(len(weights) - len(usable)),
        "index_implied_vol": index_iv,
        "avg_single_implied_vol": float(w.to_numpy() @ vols),
        "implied_correlation": average_correlation(index_iv, vols, w.to_numpy()),
        "call_put_vol_gap": skew_gap,
        "index_half_spread_volpts": index_half_spread,
        "basket_half_spread_volpts": basket_half_spread,
        "names_without_quotes": int(len(usable) - len(priced)),
        "cost_tenor_days": float(spread_table.loc[priced.to_numpy(), "days_to_expiry"].median())
        if not priced.empty
        else float("nan"),
    }


def realized_correlation_history(
    db: Any,
    start: str,
    end: str,
    *,
    window: int = 21,
    top_n: int | None = 50,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    chunk_years: int = 3,
    implied: bool = False,
    implied_days: int = 30,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per formation date in [start, end].

    Daily data is fetched `chunk_years` at a time, for the union of members over
    every formation date in the block (members as of one date would miss names
    that left mid-block), plus enough extra days for the last window.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")

    index_returns = fetch_index_returns(db, start, end).set_index("date")["sprtrn"].sort_index()
    if index_returns.empty:
        raise ValueError(f"no index returns between {start} and {end}")

    calendar = index_returns.index
    # Last trading day of each month, excluding months with no room for a full forward window.
    month_ends = pd.Series(calendar, index=calendar).resample("ME").last().dropna()
    formation_dates = [
        d for d in month_ends if calendar.get_indexer([d])[0] + window < len(calendar)
    ]

    membership = fetch_index_membership(db, start, end)
    rows: list[dict[str, Any]] = []
    # ISO strings: attrs go to JSON when the panel is written to parquet.
    skipped: list[tuple[str, str]] = []
    # Windows with a realized leg but no implied leg.
    partial: list[tuple[str, str]] = []

    for block_start in range(0, len(formation_dates), chunk_years * 12):
        block = formation_dates[block_start : block_start + chunk_years * 12]
        if not block:
            continue

        union: set[int] = set()
        for as_of in block:
            union.update(members_on(membership, as_of))

        # About 1.45 calendar days per trading day; 2 plus a week is plenty.
        buffer_days = int(window * 2.0) + 7
        fetch_start = block[0].strftime("%Y-%m-%d")
        fetch_end = (block[-1] + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")

        panel = fetch_stock_panel(db, sorted(union), fetch_start, fetch_end)
        if panel.empty:
            continue
        # pivot raises on a duplicate (date, permno); pivot_table would average it away.
        wide_returns = panel.pivot(  # noqa: PD010
            index="date", columns="permno", values="retx"
        )

        if verbose:
            print(
                f"  {block[0].date()} to {block[-1].date()}: {len(union)} names, "
                f"{len(panel):,} rows"
            )

        for as_of in block:
            # Full membership, so a member missing from the fetch is counted as dropped.
            members = members_on(membership, as_of)

            try:
                weights, dropped = formation_weights(panel, as_of, members, top_n=top_n)
            except InsufficientData as exc:
                skipped.append((as_of.strftime("%Y-%m-%d"), str(exc)))
                continue

            forward_dates = calendar[calendar > as_of][:window]
            if len(forward_dates) < window:
                skipped.append(
                    (as_of.strftime("%Y-%m-%d"), f"only {len(forward_dates)} forward trading days")
                )
                continue
            if not forward_dates.isin(wide_returns.index).all():
                n = int((~forward_dates.isin(wide_returns.index)).sum())
                skipped.append(
                    (as_of.strftime("%Y-%m-%d"), f"{n} forward date(s) absent from the panel")
                )
                continue

            try:
                measured = correlation_window(
                    wide_returns.loc[forward_dates],
                    index_returns.loc[forward_dates],
                    weights,
                    trading_days=trading_days,
                )
            except InsufficientData as exc:
                # Any other ValueError is a pipeline bug and propagates.
                skipped.append((as_of.strftime("%Y-%m-%d"), str(exc)))
                continue

            row = {
                "formation_date": as_of,
                "window_end": forward_dates[-1],
                "dropped_at_formation": dropped,
                # Compounded index return over the window, for the crash regressions in tail.py.
                "index_return": float((1.0 + index_returns.loc[forward_dates]).prod() - 1.0),
                **measured,
            }

            if implied:
                try:
                    row.update(implied_correlation_on(db, as_of, weights, days=implied_days))
                except InsufficientData as exc:
                    # Keep the realized row; every month before 1996 ends up here.
                    partial.append((as_of.strftime("%Y-%m-%d"), str(exc)))

            rows.append(row)

    if not rows:
        raise ValueError(f"no usable windows between {start} and {end}")

    history = pd.DataFrame(rows).set_index("formation_date").sort_index()
    history.attrs["skipped_windows"] = skipped
    history.attrs["windows_without_implied"] = partial

    if skipped:
        warnings.warn(
            f"{len(skipped)} of {len(skipped) + len(rows)} month-ends produced no window at all; "
            f"first: {skipped[0][0]} ({skipped[0][1]}). "
            "The full list is in the result's .attrs['skipped_windows'].",
            UserWarning,
            stacklevel=2,
        )
    if partial:
        warnings.warn(
            f"{len(partial)} of {len(rows)} windows have a realized measurement but no implied "
            f"leg; first: {partial[0][0]} ({partial[0][1]}). "
            "The full list is in the result's .attrs['windows_without_implied'].",
            UserWarning,
            stacklevel=2,
        )

    return history

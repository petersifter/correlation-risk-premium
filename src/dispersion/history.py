"""Rung 2c: the rolling history of realised correlation.

Composes the primitives from the other three modules into the measurement rung 2 exists to produce
- a monthly series of how correlated the S&P 500's members actually were, from the early 1990s to
the present, together with the diagnostics needed to say how much to trust it.

The split is the same as everywhere else. :func:`correlation_window` is pure arithmetic over one
window and is tested exhaustively on synthetic frames.
:func:`realized_correlation_history` is the driver that fetches data and loops, and is as thin as
it can be made.

How a window works
------------------
On each formation date *t* - the last trading day of a month - the basket is fixed: index
membership as of *t*, market-cap weights as of *t*. The basket is then held over the following
``window`` trading days, strictly after *t*, and volatilities are measured on that forward window.

Forming at *t* and measuring forward is the ordering a trader would actually use, and it is the
only ordering that has no lookahead. Forming at the end of a window instead - taking today's
members and measuring their volatility over the past month - silently conditions on who survived
the month. That is the single easiest way to fake this result, so it is worth being explicit that
it is not what happens here.

The size of that bias has been measured rather than assumed. Re-forming each basket at the *end* of
its window instead of the start raises mean realised correlation by **+0.0038** over a 53-month
block spanning 2008-2012, in 53 of 53 windows, and drops the incomplete-name count from 3 to 1.
Small, systematically positive, and present in every single window - the signature of genuine
lookahead. It also conceals the names that vanished, which is why the wrong version looks tidier.

The two index volatilities
--------------------------
Each window reports two, and the difference between them is a diagnostic, not noise.

``index_vol`` comes from CRSP's ``sprtrn``, the actual S&P 500 composite price return. This is the
measurement, because rung 3 compares against SPX options and those are written on the real index.

``basket_vol`` is computed from our own basket, ``R_B = sum_i w_i r_i``. Rung 1's identity holds
*exactly* against this one, so ``basket_correlation`` is a mathematical fact about our basket
rather than a measurement of the index.

``basis`` is ``index_vol - basket_vol``. It is the price of every approximation between our basket
and the real thing: a subset of names rather than all 500, market-cap weights rather than the
official float-adjusted divisor weights, and CRSP's corporate-action handling rather than S&P's.
A large basis means ``realized_correlation`` is measuring our tracking error as much as the index's
correlation, and any honest reading of the output has to look at it.

How noisy is one window
-----------------------
A 21-day estimate of average correlation has a bootstrap standard error of roughly **0.065**. For
the window formed 2010-04-30 the point estimate is 0.864 with a 95% interval of [0.707, 0.964]
(2,000 iid day resamples, 50 names).

That number governs how rung 4 may be written. Comparing implied correlation on date *t* against
realised correlation over the following month means the realised side is an estimate with an error
bar a tenth of its own range wide. No claim about the premium can rest on a single window, or on a
handful; it needs the whole panel, and any regression of realised on implied carries
errors-in-variables that attenuate the slope towards zero.

Survivorship accounting
-----------------------
Two counts come back with every window, and they are published rather than logged:

``dropped_at_formation``  members with no usable market capitalisation on *t*, so no weight.
``incomplete_in_window``  members weighted at *t* whose forward returns have holes - typically a
                          name that was acquired, delisted or halted part-way through the window.

The second is the one that matters. A name that disappears mid-window disappears for a reason, and
reasons are correlated with returns. Dropping it is the only thing you can do with an undefined
volatility, but a window where many names dropped is a window to distrust. The counts travel with
the numbers so that judgement stays available to whoever reads the output.
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
    """Measure one held window. Pure; no data access.

    ``window_returns`` is the forward window's constituent price returns, dates on the index and
    permnos on the columns. ``index_window_returns`` is the actual index return over the same
    dates. ``weights`` is the formation-date basket, indexed by permno.

    Names whose forward returns are incomplete are dropped and the remaining weights renormalised,
    because a volatility cannot be estimated from a gap. The count is returned - see the module
    docstring on why it is published rather than swallowed.
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
    # The constituent side is protected by the `usable` filter below, which keeps only columns
    # with no NaN. The index side has no such filter, and `mean()` skips NaN silently - so a single
    # hole here would annualise the index over fewer observations than the constituents and bias
    # the correlation with no error raised. Check it explicitly, with a message that says which
    # side is at fault.
    if index_window_returns.isna().any():
        n_missing = int(index_window_returns.isna().sum())
        raise ValueError(
            f"index_window_returns has {n_missing} missing observation(s) out of "
            f"{len(index_window_returns)}; volatility would be annualised over a different number "
            "of days than the constituents, biasing the correlation silently"
        )

    candidates = window_returns[weights.index]
    usable = candidates.columns[candidates.notna().all()]
    incomplete = int(len(weights) - len(usable))

    if len(usable) < 2:
        raise InsufficientData(
            f"only {len(usable)} name(s) have complete returns over the window; correlation needs "
            "two"
        )

    renormalised = weights[usable] / weights[usable].sum()
    component_returns = candidates[usable]

    component_vols = annualized_vol(component_returns, trading_days=trading_days)
    weighted_avg_vol = float(renormalised @ component_vols)

    index_vol = float(annualized_vol(index_window_returns, trading_days=trading_days))
    basket_vol = float(
        annualized_vol(basket_return(component_returns, renormalised), trading_days=trading_days)
    )

    vols = component_vols.to_numpy()
    w = renormalised.to_numpy()

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
    """Implied correlation on ``as_of``, from the same basket the realised measurement uses.

    Takes the formation-date weights computed from CRSP and prices the identical basket off the
    OptionMetrics surface, so the implied and realised numbers differ only in which volatilities go
    in. That matters: both sides inherit the same top-50 proxy approximation, so the basket error
    largely cancels in the rung 4 comparison instead of being an unmeasured wedge between them.

    Names that cannot be linked to OptionMetrics, or that have no surface on the date, are dropped
    and the remaining weights renormalised - the same rule, and the same reporting obligation, as
    the realised side.
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
        # OptionMetrics sometimes carries a surface row for SPX with a NULL implied volatility -
        # late July and early August 2020, for instance. Diagnose it here rather than letting it
        # fall through to the basket-coverage message below, which would blame the wrong thing.
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

    # Quoted half-spreads for the same basket, in volatility points, so the cost of the trade is
    # measured on the contracts actually traded rather than assumed. Names without a two-sided
    # market are excluded from the weighted cost and counted, because an untradeable leg is a
    # coverage problem rather than a free one.
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
    """Monthly realised correlation history over ``[start, end]``.

    One row per formation date - the last trading day of each month - carrying the measurements and
    the survivorship counts described in the module docstring.

    Data is fetched in ``chunk_years``-year blocks because the daily stock file is 108 million rows
    and the full member union over thirty years will not fit comfortably in memory at once. Each
    block is extended forward by a buffer so that the last formation date in it still has a
    complete forward window; the buffer is sized from ``window`` rather than guessed.

    The member union for a block is the union over *every* formation date in it, not the members as
    of one date. Fetching the latter silently omits names that left the index mid-block, which
    looks like missing data and biases the result towards survivors.
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
    # ISO strings rather than Timestamps: these ride along in DataFrame.attrs, and pandas
    # serialises attrs to JSON when writing parquet, where a Timestamp raises.
    skipped: list[tuple[str, str]] = []

    for block_start in range(0, len(formation_dates), chunk_years * 12):
        block = formation_dates[block_start : block_start + chunk_years * 12]
        if not block:
            continue

        union: set[int] = set()
        for as_of in block:
            union.update(members_on(membership, as_of))

        # Extend the fetch forward far enough that the last formation date gets a full window.
        # Calendar days per trading day is about 1.45; 2.0 plus a week is a safe margin.
        buffer_days = int(window * 2.0) + 7
        fetch_start = block[0].strftime("%Y-%m-%d")
        fetch_end = (block[-1] + pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")

        panel = fetch_stock_panel(db, sorted(union), fetch_start, fetch_end)
        if panel.empty:
            continue
        # pivot, not pivot_table: pivot raises on duplicate (date, permno) rows whereas
        # pivot_table would silently average them. CRSP should never have duplicates, and if it
        # does that is a data defect worth failing on rather than smoothing away.
        wide_returns = panel.pivot(  # noqa: PD010
            index="date", columns="permno", values="retx"
        )

        if verbose:
            print(
                f"  {block[0].date()} to {block[-1].date()}: {len(union)} names, "
                f"{len(panel):,} rows"
            )

        for as_of in block:
            # Pass the full membership rather than pre-filtering against the panel's columns: a
            # member absent from the fetch entirely must be counted by formation_weights as a drop,
            # not silently removed before any counter sees it.
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
                # Only an expected shortage is skipped. A plain ValueError from here means a
                # pipeline defect - a name absent from the fetch, misaligned dates, a hole in the
                # index series - and is allowed to propagate rather than silently cost a window.
                skipped.append((as_of.strftime("%Y-%m-%d"), str(exc)))
                continue

            row = {
                "formation_date": as_of,
                "window_end": forward_dates[-1],
                "dropped_at_formation": dropped,
                **measured,
            }

            if implied:
                # Priced off the same basket as the realised leg, so the two differ only in which
                # volatilities go in. A date with no usable surface loses only the implied columns;
                # the realised measurement for that window is still valid and is kept.
                try:
                    row.update(implied_correlation_on(db, as_of, weights, days=implied_days))
                except InsufficientData as exc:
                    skipped.append((as_of.strftime("%Y-%m-%d"), f"implied leg unavailable: {exc}"))

            rows.append(row)

    if not rows:
        raise ValueError(f"no usable windows between {start} and {end}")

    history = pd.DataFrame(rows).set_index("formation_date").sort_index()
    history.attrs["skipped_windows"] = skipped

    # A silently short history is indistinguishable from a correct one, so say so. Every skip here
    # is an expected shortage; pipeline defects raise instead of landing in this list.
    if skipped:
        warnings.warn(
            f"{len(skipped)} of {len(skipped) + len(rows)} month-ends produced no window; "
            f"first: {skipped[0][0]} ({skipped[0][1]}). "
            "The full list is in the result's .attrs['skipped_windows'].",
            UserWarning,
            stacklevel=2,
        )

    return history

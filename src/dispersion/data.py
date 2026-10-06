"""WRDS queries and the point-in-time basket.

The `fetch_*` functions are thin SQL wrappers and need a WRDS login.
`members_on` and `formation_weights` are pure pandas and are tested on
synthetic frames.

Baskets are formed on the formation date and held: index membership and
market-cap weights are read as of that date and nothing after it. Constituents
use `retx` and the index uses `sprtrn`, both price returns, because SPX options
are written on the price index. Names without a usable market cap are dropped,
the remaining weights renormalized, and the number dropped is returned so it can
be reported.

Tables (CRSP Annual Update):

    crsp.dsp500list   permno, start, ending
    crsp.dsf          date, permno, ret, retx, prc, shrout
    crsp.dsi          date, sprtrn, vwretd

CRSP negates prc when it is a bid/ask midpoint, so market cap is
abs(prc) * shrout, in $ thousands.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_TABLES",
    "InsufficientData",
    "connect",
    "fetch_atm_implied_vols",
    "fetch_atm_quote_spreads",
    "fetch_index_membership",
    "fetch_index_returns",
    "fetch_optionmetrics_link",
    "fetch_stock_panel",
    "formation_weights",
    "members_on",
    "query",
    "verify_schema",
]


class InsufficientData(ValueError):
    """Not enough usable names to measure a window.

    The driver in history.py skips windows that raise this and lets any other
    ValueError (misaligned dates, a name missing from a fetch) propagate.
    """


SPX_SECID = 108105
"""OptionMetrics secid of the S&P 500 index (from optionm.secnmd)."""

REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "crsp.dsp500list": ("permno", "start", "ending"),
    "crsp.dsf": ("date", "permno", "ret", "retx", "prc", "shrout"),
    "crsp.dsi": ("date", "sprtrn", "vwretd"),
    "wrdsapps_link_crsp_optionm.opcrsphist": ("secid", "permno", "sdate", "edate", "score"),
}


def connect(username: str) -> Any:
    """Open a WRDS connection.

    The password comes from a .pgpass file, created once with
    `db.create_pgpass_file()`. Without it the client prompts.
    """
    import wrds  # lazy, so the pure functions don't need it

    return wrds.Connection(wrds_username=username)


def verify_schema(db: Any) -> pd.DataFrame:
    """One row per table in REQUIRED_TABLES: reachable or not, and missing columns."""
    rows = []
    for qualified, required in REQUIRED_TABLES.items():
        library, table = qualified.split(".")
        try:
            available = {c.lower() for c in db.describe_table(library, table)["name"]}
            missing = tuple(sorted(set(required) - available))
            rows.append({"table": qualified, "reachable": True, "missing_columns": missing})
        except Exception as exc:
            rows.append({"table": qualified, "reachable": False, "missing_columns": (str(exc),)})
    return pd.DataFrame(rows)


def query(
    db: Any,
    sql: str,
    params: dict[str, Any] | None = None,
    date_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run `sql` on WRDS and return a DataFrame.

    `db.raw_sql` passes pandas a SQLAlchemy 1.4 connection, which pandas 3
    rejects ('Connection' object has no attribute 'cursor'), and wrds pins
    sqlalchemy<2. This uses the psycopg2 connection underneath instead. It is
    pooled, so it isn't closed here. Parameters use psycopg2's %(name)s style;
    a tuple becomes a SQL list, so `IN %(permnos)s` works.
    """
    # pandas warns on every call that a raw DBAPI connection is untested.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable",
            category=UserWarning,
        )
        return pd.read_sql_query(
            sql,
            db.connection.connection,
            params=params,
            parse_dates=date_cols,
        )


def fetch_index_membership(db: Any, start: str, end: str) -> pd.DataFrame:
    """S&P 500 membership spells overlapping [start, end]: permno, from_date, thru_date.

    A name still in the index has a null `ending`; it is set to 2262-01-01.
    """
    sql = """
        SELECT permno, start AS from_date, ending AS thru_date
        FROM crsp.dsp500list
        WHERE (ending IS NULL OR ending >= %(start)s)
          AND start <= %(end)s
    """
    membership = query(
        db, sql, params={"start": start, "end": end}, date_cols=["from_date", "thru_date"]
    )
    membership["thru_date"] = membership["thru_date"].fillna(pd.Timestamp("2262-01-01"))
    return membership.astype({"permno": "int64"})


def fetch_stock_panel(db: Any, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """Daily ret, retx and market cap for `permnos` over [start, end].

    retx (price return) is what the volatility uses; ret is kept for comparison.
    """
    sql = """
        SELECT date, permno, ret, retx, ABS(prc) * shrout AS market_cap
        FROM crsp.dsf
        WHERE date BETWEEN %(start)s AND %(end)s
          AND permno IN %(permnos)s
    """
    panel = query(
        db,
        sql,
        params={"start": start, "end": end, "permnos": tuple(permnos)},
        date_cols=["date"],
    )
    return panel.astype({"permno": "int64"})


def fetch_index_returns(db: Any, start: str, end: str) -> pd.DataFrame:
    """Daily sprtrn (S&P 500 price return) and vwretd (CRSP value-weighted total return)."""
    sql = """
        SELECT date, sprtrn, vwretd
        FROM crsp.dsi
        WHERE date BETWEEN %(start)s AND %(end)s
    """
    return query(db, sql, params={"start": start, "end": end}, date_cols=["date"])


def members_on(membership: pd.DataFrame, as_of: pd.Timestamp) -> pd.Index:
    """Permnos in the index on `as_of`. Spells include both endpoints."""
    required = {"permno", "from_date", "thru_date"}
    if not required.issubset(membership.columns):
        raise ValueError(f"membership must have columns {sorted(required)}")

    in_index = (membership["from_date"] <= as_of) & (membership["thru_date"] >= as_of)
    return pd.Index(membership.loc[in_index, "permno"].unique(), name="permno").sort_values()


def formation_weights(
    stock_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    permnos: pd.Index,
    *,
    top_n: int | None = None,
) -> tuple[pd.Series, int]:
    """Market-cap weights on `as_of`, renormalized to sum to 1.

    Returns the weights and how many of `permnos` had no usable market cap that
    day. `top_n` keeps the largest n names; None keeps all of them. Only the
    `as_of` cross-section is used.
    """
    on_date = stock_panel.loc[stock_panel["date"] == as_of, ["permno", "market_cap"]]
    caps = (
        on_date.set_index("permno")["market_cap"]
        .reindex(permnos)
        .replace([np.inf, -np.inf], np.nan)
    )
    caps = caps.where(caps > 0.0)

    dropped = int(caps.isna().sum())
    caps = caps.dropna()

    if caps.empty:
        raise InsufficientData(f"no constituent has a usable market cap on {as_of.date()}")
    if top_n is not None:
        caps = caps.nlargest(top_n)
    if caps.size < 2:
        raise InsufficientData(
            f"only {caps.size} constituent(s) usable on {as_of.date()}; correlation needs two"
        )

    return caps / caps.sum(), dropped


def fetch_optionmetrics_link(db: Any, permnos: list[int], as_of: str) -> pd.Series:
    """permno -> secid as of `as_of`, using score = 1 links only.

    At the seven dates checked (1996, 2000, 2005, 2010, 2015, 2020, 2023) score 1
    covered all 50 basket names. The join is dated because both ids get reused.
    """
    link = query(
        db,
        """
        SELECT permno, secid
        FROM wrdsapps_link_crsp_optionm.opcrsphist
        WHERE sdate <= %(as_of)s AND edate >= %(as_of)s
          AND score = 1
          AND permno IN %(permnos)s
        """,
        params={"as_of": as_of, "permnos": tuple(int(p) for p in permnos)},
    )
    if link.empty:
        return pd.Series(dtype="int64", name="secid")

    mapped = link.drop_duplicates("permno").set_index("permno")["secid"].astype("int64")

    # Two permnos on one secid would misalign vols and weights downstream. None today, but check.
    duplicated = mapped[mapped.duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"{duplicated.nunique()} secid(s) map to more than one permno on {as_of}: "
            f"{sorted(duplicated.unique())[:3]}. Resolve the link before pricing the basket"
        )
    return mapped


def fetch_atm_implied_vols(
    db: Any,
    secids: list[int],
    as_of: str,
    *,
    days: int = 30,
) -> pd.DataFrame:
    """30-day at-the-money implied vols from the OptionMetrics volatility surface.

    One row per secid: the 50-delta call vol, the -50-delta put vol, and their
    mean in `atm_vol`. The two sit at slightly different strikes, so the gap is
    local skew (about 0.8 vol points across the basket on 2019-11-29, 8.0 on
    2020-02-28). 30 calendar days matches the 21-trading-day realized window.
    The surface tables are split by year, so `as_of` picks the table.
    """
    year = pd.Timestamp(as_of).year
    surface = query(
        db,
        f"""
        SELECT secid, cp_flag, impl_volatility
        FROM optionm.vsurfd{year}
        WHERE date = %(as_of)s
          AND days = %(days)s
          AND ABS(delta) = 50
          AND secid IN %(secids)s
        """,
        params={"as_of": as_of, "days": days, "secids": tuple(int(s) for s in secids)},
    )
    if surface.empty:
        return pd.DataFrame(columns=["call_vol", "put_vol", "atm_vol"], dtype=float)

    # pivot_table averages the rare duplicate surface row.
    wide = surface.pivot_table(
        index="secid", columns="cp_flag", values="impl_volatility", aggfunc="mean"
    ).rename(columns={"C": "call_vol", "P": "put_vol"})
    for leg in ("call_vol", "put_vol"):
        if leg not in wide.columns:
            wide[leg] = float("nan")
    wide["atm_vol"] = wide[["call_vol", "put_vol"]].mean(axis=1)
    wide.index = wide.index.astype("int64")
    return wide[["call_vol", "put_vol", "atm_vol"]]


def fetch_atm_quote_spreads(
    db: Any,
    secids: list[int],
    as_of: str,
    *,
    target_days: int = 30,
    days_low: int = 10,
    days_high: int = 60,
    delta_low: float = 0.40,
    delta_high: float = 0.60,
) -> pd.DataFrame:
    """Quoted half-spread of near-the-money options in volatility points.

        half_spread_volpts = 100 * (best_offer - best_bid) / (2 * vega)

    OptionMetrics vega is per 1.00 of volatility (an SPX vega of 359.96 matched
    0.3989 * S * sqrt(T) to within 0.3%), hence the 100.

    The expiry window is wide because before weeklies (about 2012) the listed
    expiries nearest a month end were roughly 15 and 50 days out, and a 25-35
    day window found nothing. For each secid only the expiry closest to
    `target_days` is kept, and its tenor is returned in `days_to_expiry`.
    Quotes with a zero bid or non-positive vega are dropped. Names with no
    usable quote are absent from the result.
    """
    year = pd.Timestamp(as_of).year
    sql = f"""
        SELECT secid,
               exdate - date AS days_to_expiry,
               100.0 * (best_offer - best_bid) / (2.0 * vega) AS half_spread_volpts
        FROM optionm.opprcd{year}
        WHERE date = %(as_of)s
          AND secid IN %(secids)s
          AND exdate - date BETWEEN %(days_low)s AND %(days_high)s
          AND ABS(delta) BETWEEN %(delta_low)s AND %(delta_high)s
          AND vega > 0
          AND best_bid > 0
          AND best_offer > best_bid
    """

    quotes = query(
        db,
        sql,
        params={
            "as_of": as_of,
            "secids": tuple(int(s) for s in secids),
            "days_low": days_low,
            "days_high": days_high,
            "delta_low": delta_low,
            "delta_high": delta_high,
        },
    )
    if quotes.empty:
        return pd.DataFrame(columns=["half_spread_volpts", "days_to_expiry"], dtype=float)

    # Nearest expiry per name, then the median over the near-the-money strikes there.
    quotes["distance"] = (quotes["days_to_expiry"] - target_days).abs()
    nearest = quotes.loc[quotes.groupby("secid")["distance"].idxmin(), ["secid", "days_to_expiry"]]
    chosen = quotes.merge(nearest, on=["secid", "days_to_expiry"], how="inner")

    result = chosen.groupby("secid").agg(
        half_spread_volpts=("half_spread_volpts", "median"),
        days_to_expiry=("days_to_expiry", "first"),
    )
    result.index = result.index.astype("int64")
    return result

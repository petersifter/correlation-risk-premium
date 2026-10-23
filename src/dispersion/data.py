from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


class InsufficientData(ValueError):
    """Too few usable names to measure a window; the driver skips these."""


SPX_SECID = 108105
"""OptionMetrics secid of the S&P 500 index (from optionm.secnmd)."""

REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "crsp.dsp500list": ("permno", "start", "ending"),
    "crsp.dsf": ("date", "permno", "ret", "retx", "prc", "shrout"),
    "crsp.dsi": ("date", "sprtrn", "vwretd"),
    "wrdsapps_link_crsp_optionm.opcrsphist": ("secid", "permno", "sdate", "edate", "score"),
}


def connect(username: str) -> Any:
    import wrds

    return wrds.Connection(wrds_username=username)


def verify_schema(db: Any) -> pd.DataFrame:
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
    """db.raw_sql fails under pandas 3 (wrds pins sqlalchemy<2), so use the DBAPI connection."""
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
    sql = """
        SELECT date, sprtrn, vwretd
        FROM crsp.dsi
        WHERE date BETWEEN %(start)s AND %(end)s
    """
    return query(db, sql, params={"start": start, "end": end}, date_cols=["date"])


def members_on(membership: pd.DataFrame, as_of: pd.Timestamp) -> pd.Index:
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
            f"only {caps.size} usable constituent(s) on {as_of.date()}, correlation needs two"
        )

    return caps / caps.sum(), dropped


def fetch_optionmetrics_link(db: Any, permnos: list[int], as_of: str) -> pd.Series:
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

    duplicated = mapped[mapped.duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"{duplicated.nunique()} secid(s) map to more than one permno on {as_of}: "
            f"{sorted(duplicated.unique())[:3]}"
        )
    return mapped


def fetch_atm_implied_vols(
    db: Any,
    secids: list[int],
    as_of: str,
    *,
    days: int = 30,
) -> pd.DataFrame:
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
    year = pd.Timestamp(as_of).year
    # OptionMetrics vega is per 1.00 of volatility, so x100 gives vol points
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

    quotes["distance"] = (quotes["days_to_expiry"] - target_days).abs()
    nearest = quotes.loc[quotes.groupby("secid")["distance"].idxmin(), ["secid", "days_to_expiry"]]
    chosen = quotes.merge(nearest, on=["secid", "days_to_expiry"], how="inner")

    result = chosen.groupby("secid").agg(
        half_spread_volpts=("half_spread_volpts", "median"),
        days_to_expiry=("days_to_expiry", "first"),
    )
    result.index = result.index.astype("int64")
    return result

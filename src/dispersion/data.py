"""CRSP data access and panel assembly for rung 2.

Split deliberately in two halves:

* The ``fetch_*`` functions issue SQL against WRDS. They cannot be tested without a licensed
  subscription, so they are kept as thin as possible - a query, a rename, a type coercion, nothing
  that could be wrong in an interesting way.
* Everything else is pure pandas over frames that happen to have come from CRSP. That half carries
  all of the research decisions, and ``tests/test_data.py`` exercises it exhaustively on synthetic
  frames shaped like CRSP output.

The three decisions in this module
----------------------------------
**1. The basket is formed at the start of each window.** Taking today's index members and measuring
their volatility over the past month uses knowledge of who survived the month. A trader forms a
basket and then holds it. So membership and weights are read as of the window's first date and held
through it. :func:`members_on` and :func:`formation_weights` are both as-of functions for this
reason, and neither of them can see past its ``as_of`` argument.

**2. Price returns on both legs.** SPX options are written on the S&P 500 *price* index, which
excludes dividends. CRSP offers ``ret`` (with dividends) and ``retx`` (without). Measuring index
volatility from a price return while measuring constituent volatility from total returns compares
two different quantities, and the resulting correlation is biased. So ``retx`` for constituents and
``sprtrn`` for the index - matched price returns. ``ret`` is fetched too, so the sensitivity of the
result to this choice can be reported rather than asserted.

**3. Names with incomplete data are dropped and the remainder renormalised**, with the count
reported per window. A name that halts or delists part-way through a window has no usable
volatility estimate. Dropping it is correct; dropping it *silently* is how survivorship bias gets
in. :func:`formation_weights` returns the weights and the caller is expected to carry the drop
count into the diagnostics - see ``reports/`` once rung 2c has been run.

Schema
------
Written against the classic CRSP layout under the Annual Update vintage:

====================  ==========================================================
``crsp.dsp500list``   S&P 500 membership: ``permno``, ``start``, ``ending``
``crsp.dsf``          daily stock file: ``date``, ``permno``, ``ret``, ``retx``,
                      ``prc``, ``shrout``
``crsp.dsi``          daily index file: ``date``, ``sprtrn``, ``vwretd``
====================  ==========================================================

``prc`` is negated by CRSP when it is a bid/ask midpoint rather than a close, so market
capitalisation uses ``abs(prc)``. ``shrout`` is in thousands; the absolute scale is irrelevant to
weights but the unit is recorded here so nobody multiplies it twice.

CRSP's newer CIZ format renames these tables. :func:`verify_schema` checks what is actually
reachable with the caller's credentials before any real query is issued, so a rename surfaces as a
clear error rather than a confusing empty frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_TABLES",
    "connect",
    "fetch_index_membership",
    "fetch_index_returns",
    "fetch_stock_panel",
    "formation_weights",
    "members_on",
    "query",
    "verify_schema",
]

REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "crsp.dsp500list": ("permno", "start", "ending"),
    "crsp.dsf": ("date", "permno", "ret", "retx", "prc", "shrout"),
    "crsp.dsi": ("date", "sprtrn", "vwretd"),
}


def connect(username: str) -> Any:
    """Open a WRDS connection.

    Credentials are never handled here. WRDS caches them in a ``.pgpass`` file
    (``%APPDATA%\\postgresql\\pgpass.conf`` on Windows) which the user creates once, interactively,
    by calling ``db.create_pgpass_file()``. Without that file this call will try to prompt, which
    is useless in a non-interactive context - so if it hangs or raises, the fix is to run the
    interactive setup by hand, not to put a password in this repository.
    """
    import wrds  # imported lazily so the rest of the module is usable without the dependency

    return wrds.Connection(wrds_username=username)


def verify_schema(db: Any) -> pd.DataFrame:
    """Check that every table and column in :data:`REQUIRED_TABLES` is actually reachable.

    Returns one row per required table with the columns that are missing, if any. Run this before
    the first real query: a subscription gap or a CIZ rename otherwise shows up as an empty frame
    several steps later, which is a much worse error to debug.
    """
    rows = []
    for qualified, required in REQUIRED_TABLES.items():
        library, table = qualified.split(".")
        try:
            available = {c.lower() for c in db.describe_table(library, table)["name"]}
            missing = tuple(sorted(set(required) - available))
            rows.append({"table": qualified, "reachable": True, "missing_columns": missing})
        except Exception as exc:
            # Any failure at all means "not usable with these credentials"; the message says why.
            rows.append({"table": qualified, "reachable": False, "missing_columns": (str(exc),)})
    return pd.DataFrame(rows)


def query(
    db: Any,
    sql: str,
    params: dict[str, Any] | None = None,
    date_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run ``sql`` against WRDS and return a DataFrame. Use this instead of ``db.raw_sql``.

    ``wrds.Connection.raw_sql`` hands a SQLAlchemy 1.4 ``Connection`` to
    ``pandas.read_sql_query``. pandas 3 no longer recognises that object, falls back to treating it
    as a raw DBAPI connection, and fails with ``AttributeError: 'Connection' object has no
    attribute 'cursor'``. Upgrading SQLAlchemy is not available as a fix because ``wrds`` pins
    ``sqlalchemy<2``.

    So reach through to the DBAPI connection that SQLAlchemy is wrapping, which pandas does
    support. The connection is pooled, so this borrows it rather than owning it - do not close it
    here. Note that psycopg2's parameter style is ``%(name)s``, which is what every query in this
    module is written in; a tuple parameter adapts to a SQL list, so ``IN %(permnos)s`` works.
    """
    return pd.read_sql_query(
        sql,
        db.connection.connection,
        params=params,
        parse_dates=date_cols,
    )


def fetch_index_membership(db: Any, start: str, end: str) -> pd.DataFrame:
    """S&P 500 membership spells overlapping ``[start, end]``.

    Returns ``permno``, ``from_date``, ``thru_date``. An open-ended spell (a name still in the
    index) has a null ``ending`` in CRSP, which is replaced by a far-future date so that interval
    arithmetic does not need to special-case it.
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
    """Daily returns and market capitalisation for ``permnos`` over ``[start, end]``.

    ``retx`` is the price return and the one used for volatility - see decision 2 in the module
    docstring. ``ret`` is returned alongside it so the sensitivity to that choice can be measured.
    ``market_cap`` is ``abs(prc) * shrout`` in thousands of dollars; CRSP negates ``prc`` when the
    figure is a bid/ask midpoint rather than a traded close.
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
    """Daily index returns over ``[start, end]``.

    ``sprtrn`` is the S&P 500 composite *price* return and is the series to use, because SPX
    options are written on the price index. ``vwretd`` - the CRSP value-weighted total return - is
    returned alongside it as a cross-check on the basket construction, not as the measurement.
    """
    sql = """
        SELECT date, sprtrn, vwretd
        FROM crsp.dsi
        WHERE date BETWEEN %(start)s AND %(end)s
    """
    return query(db, sql, params={"start": start, "end": end}, date_cols=["date"])


def members_on(membership: pd.DataFrame, as_of: pd.Timestamp) -> pd.Index:
    """Permnos in the index on ``as_of``, from membership spells.

    A spell is inclusive at both ends: a name that entered on ``as_of`` is in the index that day.
    Point-in-time by construction - this function cannot see anything after ``as_of``, which is
    what keeps decision 1 honest.
    """
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
    """Market-cap weights as of ``as_of``, renormalised to sum to 1.

    Returns the weights and the number of requested names that had no usable market
    capitalisation on that date. That count is a diagnostic, not a detail: it is the measure of how
    much survivorship risk the window carries, and it belongs in the published output.

    ``top_n`` keeps only the largest *n* names by market capitalisation and renormalises. Real
    dispersion books trade a proxy basket of the largest 50 or so rather than all 500, because the
    small tail contributes almost no vega and a great deal of transaction cost. Passing ``None``
    uses every member.

    Weights are computed from a single day's cross-section - the formation date - so this function,
    like :func:`members_on`, cannot see into the window it is forming.
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
        raise ValueError(f"no constituent has a usable market capitalisation on {as_of.date()}")
    if top_n is not None:
        caps = caps.nlargest(top_n)
    if caps.size < 2:
        raise ValueError(
            f"only {caps.size} constituent(s) usable on {as_of.date()}; correlation needs two"
        )

    return caps / caps.sum(), dropped

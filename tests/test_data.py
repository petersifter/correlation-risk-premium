"""Tests for rung 2c's assembly logic.

The ``fetch_*`` functions are not tested here - they need a licensed WRDS subscription, and they
are deliberately thin enough that there is nothing in them to get wrong beyond the SQL itself.
What is tested is every decision described in ``data.py``'s module docstring, because those are the
parts that would silently bias a result.

The fixtures are shaped like CRSP output on purpose: integer permnos rather than tickers, a long
(date, permno) panel rather than a wide one, market capitalisation already multiplied out, and
membership expressed as spells with inclusive endpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.data import REQUIRED_TABLES, formation_weights, members_on


@pytest.fixture
def membership() -> pd.DataFrame:
    """Four names with overlapping index spells, including one still in the index."""
    return pd.DataFrame(
        {
            "permno": [10001, 10002, 10003, 10004],
            "from_date": pd.to_datetime(["2000-01-01", "2000-01-01", "2005-06-15", "2010-03-01"]),
            "thru_date": pd.to_datetime(["2008-12-31", "2262-01-01", "2012-01-31", "2262-01-01"]),
        }
    )


@pytest.fixture
def stock_panel() -> pd.DataFrame:
    """A two-date long panel with a deliberate hole and a deliberate zero."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2006-01-03"] * 4 + ["2006-01-04"] * 4,
            ),
            "permno": [10001, 10002, 10003, 10004] * 2,
            "retx": [0.01, -0.02, 0.005, 0.0, 0.002, 0.01, -0.01, 0.0],
            "market_cap": [400_000.0, 300_000.0, 200_000.0, np.nan, 410_000.0, 0.0, 190_000.0, 1.0],
        }
    )


def test_required_tables_are_declared_for_schema_verification():
    """verify_schema is only useful if the contract it checks is explicit."""
    assert "crsp.dsf" in REQUIRED_TABLES
    assert "retx" in REQUIRED_TABLES["crsp.dsf"]
    assert "sprtrn" in REQUIRED_TABLES["crsp.dsi"]


def test_members_on_resolves_a_mid_sample_date(membership):
    """On 2006-01-03 three names are in: 10001, 10002 and 10003. 10004 has not joined yet."""
    members = members_on(membership, pd.Timestamp("2006-01-03"))
    assert list(members) == [10001, 10002, 10003]


def test_members_on_excludes_names_after_their_spell_ends(membership):
    """10001 leaves at the end of 2008, so by 2009 only 10002 and 10003 are still in."""
    assert list(members_on(membership, pd.Timestamp("2009-01-02"))) == [10002, 10003]


def test_membership_spells_are_inclusive_at_both_ends(membership):
    """A name that joins on a date is in the index that date, and likewise for its last day."""
    assert 10003 in members_on(membership, pd.Timestamp("2005-06-15"))
    assert 10003 not in members_on(membership, pd.Timestamp("2005-06-14"))
    assert 10001 in members_on(membership, pd.Timestamp("2008-12-31"))
    assert 10001 not in members_on(membership, pd.Timestamp("2009-01-01"))


def test_open_ended_spells_are_still_in_the_index(membership):
    """10002 and 10004 never leave, so a late date must still find them."""
    assert list(members_on(membership, pd.Timestamp("2020-07-01"))) == [10002, 10004]


def test_members_on_is_point_in_time(membership):
    """The whole no-lookahead guarantee: the result must not depend on later spells.

    Truncating the membership table to spells that had already started cannot change what
    members_on reports for an earlier date.
    """
    as_of = pd.Timestamp("2006-01-03")
    truncated = membership[membership["from_date"] <= as_of]
    assert list(members_on(membership, as_of)) == list(members_on(truncated, as_of))


def test_members_on_rejects_a_malformed_membership_frame():
    with pytest.raises(ValueError, match="must have columns"):
        members_on(pd.DataFrame({"permno": [1]}), pd.Timestamp("2006-01-03"))


def test_formation_weights_are_cap_weighted_and_sum_to_one(stock_panel, membership):
    """Caps of 400k, 300k and 200k give weights of 4/9, 3/9 and 2/9."""
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    weights, dropped = formation_weights(stock_panel, as_of, permnos)

    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc[10001] == pytest.approx(4 / 9)
    assert weights.loc[10002] == pytest.approx(3 / 9)
    assert weights.loc[10003] == pytest.approx(2 / 9)
    assert dropped == 0


def test_names_without_a_usable_market_cap_are_dropped_and_counted(stock_panel, membership):
    """10004 has a NaN cap on 2006-01-03; it is excluded and the drop is reported.

    The count is the point. Dropping is correct; dropping silently is how survivorship bias gets
    into a result that otherwise looks clean.
    """
    as_of = pd.Timestamp("2006-01-03")
    permnos = pd.Index([10001, 10002, 10003, 10004], name="permno")

    weights, dropped = formation_weights(stock_panel, as_of, permnos)

    assert 10004 not in weights.index
    assert dropped == 1
    assert weights.sum() == pytest.approx(1.0)


def test_zero_market_cap_is_treated_as_missing(stock_panel):
    """A zero cap is a data defect, not a name with no weight; it must not divide into anything."""
    as_of = pd.Timestamp("2006-01-04")
    permnos = pd.Index([10001, 10002, 10003], name="permno")

    weights, dropped = formation_weights(stock_panel, as_of, permnos)

    assert 10002 not in weights.index
    assert dropped == 1
    assert weights.sum() == pytest.approx(1.0)


def test_top_n_keeps_the_largest_names_and_renormalises(stock_panel, membership):
    """Proxy dispersion: the largest two of 400k/300k/200k are 10001 and 10002, at 4/7 and 3/7."""
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    weights, _ = formation_weights(stock_panel, as_of, permnos, top_n=2)

    assert list(weights.index) == [10001, 10002]
    assert weights.loc[10001] == pytest.approx(4 / 7)
    assert weights.loc[10002] == pytest.approx(3 / 7)
    assert weights.sum() == pytest.approx(1.0)


def test_top_n_larger_than_the_universe_is_harmless(stock_panel, membership):
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    weights, _ = formation_weights(stock_panel, as_of, permnos, top_n=500)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.size == 3


def test_formation_weights_use_only_the_formation_date(stock_panel, membership):
    """No lookahead: later rows of the panel cannot change the weights on the formation date."""
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    full, _ = formation_weights(stock_panel, as_of, permnos)
    truncated_panel = stock_panel[stock_panel["date"] <= as_of]
    truncated, _ = formation_weights(truncated_panel, as_of, permnos)

    pd.testing.assert_series_equal(full, truncated)


def test_a_date_with_no_usable_caps_raises(stock_panel):
    with pytest.raises(ValueError, match="no constituent has a usable market capitalisation"):
        formation_weights(stock_panel, pd.Timestamp("1990-01-02"), pd.Index([10001, 10002]))


def test_a_single_usable_name_raises(stock_panel):
    """Correlation is undefined with one name, so this must fail loudly rather than return 1.0."""
    as_of = pd.Timestamp("2006-01-03")
    with pytest.raises(ValueError, match="correlation needs two"):
        formation_weights(stock_panel, as_of, pd.Index([10001, 10004]))

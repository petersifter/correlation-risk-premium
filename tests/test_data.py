from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.data import REQUIRED_TABLES, formation_weights, members_on


@pytest.fixture
def membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permno": [10001, 10002, 10003, 10004],
            "from_date": pd.to_datetime(["2000-01-01", "2000-01-01", "2005-06-15", "2010-03-01"]),
            "thru_date": pd.to_datetime(["2008-12-31", "2262-01-01", "2012-01-31", "2262-01-01"]),
        }
    )


@pytest.fixture
def stock_panel() -> pd.DataFrame:
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


def test_required_tables():
    assert "crsp.dsf" in REQUIRED_TABLES
    assert "retx" in REQUIRED_TABLES["crsp.dsf"]
    assert "sprtrn" in REQUIRED_TABLES["crsp.dsi"]


def test_members_mid_sample(membership):
    members = members_on(membership, pd.Timestamp("2006-01-03"))
    assert list(members) == [10001, 10002, 10003]


def test_members_after_spell_ends(membership):
    assert list(members_on(membership, pd.Timestamp("2009-01-02"))) == [10002, 10003]


def test_spells_inclusive(membership):
    assert 10003 in members_on(membership, pd.Timestamp("2005-06-15"))
    assert 10003 not in members_on(membership, pd.Timestamp("2005-06-14"))
    assert 10001 in members_on(membership, pd.Timestamp("2008-12-31"))
    assert 10001 not in members_on(membership, pd.Timestamp("2009-01-01"))


def test_open_spells(membership):
    assert list(members_on(membership, pd.Timestamp("2020-07-01"))) == [10002, 10004]


def test_members_ignore_later_spells(membership):
    as_of = pd.Timestamp("2006-01-03")
    truncated = membership[membership["from_date"] <= as_of]
    assert list(members_on(membership, as_of)) == list(members_on(truncated, as_of))


def test_members_bad_frame():
    with pytest.raises(ValueError, match="must have columns"):
        members_on(pd.DataFrame({"permno": [1]}), pd.Timestamp("2006-01-03"))


def test_cap_weights(stock_panel, membership):
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    weights, dropped = formation_weights(stock_panel, as_of, permnos)

    assert weights.sum() == pytest.approx(1.0)
    assert weights.loc[10001] == pytest.approx(4 / 9)
    assert weights.loc[10002] == pytest.approx(3 / 9)
    assert weights.loc[10003] == pytest.approx(2 / 9)
    assert dropped == 0


def test_missing_cap_dropped_and_counted(stock_panel, membership):
    as_of = pd.Timestamp("2006-01-03")
    permnos = pd.Index([10001, 10002, 10003, 10004], name="permno")

    weights, dropped = formation_weights(stock_panel, as_of, permnos)

    assert 10004 not in weights.index
    assert dropped == 1
    assert weights.sum() == pytest.approx(1.0)


def test_zero_cap_dropped(stock_panel):
    as_of = pd.Timestamp("2006-01-04")
    permnos = pd.Index([10001, 10002, 10003], name="permno")

    weights, dropped = formation_weights(stock_panel, as_of, permnos)

    assert 10002 not in weights.index
    assert dropped == 1
    assert weights.sum() == pytest.approx(1.0)


def test_top_n(stock_panel, membership):
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    weights, _ = formation_weights(stock_panel, as_of, permnos, top_n=2)

    assert list(weights.index) == [10001, 10002]
    assert weights.loc[10001] == pytest.approx(4 / 7)
    assert weights.loc[10002] == pytest.approx(3 / 7)
    assert weights.sum() == pytest.approx(1.0)


def test_top_n_above_universe(stock_panel, membership):
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    weights, _ = formation_weights(stock_panel, as_of, permnos, top_n=500)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.size == 3


def test_weights_ignore_later_rows(stock_panel, membership):
    as_of = pd.Timestamp("2006-01-03")
    permnos = members_on(membership, as_of)

    full, _ = formation_weights(stock_panel, as_of, permnos)
    truncated_panel = stock_panel[stock_panel["date"] <= as_of]
    truncated, _ = formation_weights(truncated_panel, as_of, permnos)

    pd.testing.assert_series_equal(full, truncated)


def test_no_usable_caps(stock_panel):
    with pytest.raises(ValueError, match="no constituent has a usable market cap"):
        formation_weights(stock_panel, pd.Timestamp("1990-01-02"), pd.Index([10001, 10002]))


def test_one_usable_name(stock_panel):
    as_of = pd.Timestamp("2006-01-03")
    with pytest.raises(ValueError, match="correlation needs two"):
        formation_weights(stock_panel, as_of, pd.Index([10001, 10004]))

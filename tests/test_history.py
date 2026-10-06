"""Tests for rung 2c's window measurement.

``realized_correlation_history`` is the driver and needs WRDS, so it is not tested here.
``correlation_window`` carries every decision that could bias the output and is tested in full.

The important one is ``test_basket_correlation_is_exact_against_the_full_matrix``. Because
volatilities are not demeaned, the identity holds exactly against the basket we construct
ourselves, so ``basket_correlation`` can be checked with zero tolerance. ``realized_correlation``
cannot - it is measured against the real index and therefore carries the basis - which is exactly
why both are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.correlation import average_correlation, index_variance
from dispersion.data import InsufficientData
from dispersion.history import correlation_window
from dispersion.realized import TRADING_DAYS_PER_YEAR

WINDOW = 21
PERMNOS = [10001, 10002, 10003, 10004]


@pytest.fixture
def window_returns() -> pd.DataFrame:
    """A complete forward window of price returns for four names."""
    rng = np.random.default_rng(99)
    corr = np.full((4, 4), 0.4)
    np.fill_diagonal(corr, 1.0)
    draws = rng.multivariate_normal(np.zeros(4), corr * 0.015**2, size=WINDOW)
    dates = pd.bdate_range("2020-03-02", periods=WINDOW)
    return pd.DataFrame(draws, index=dates, columns=PERMNOS)


@pytest.fixture
def weights() -> pd.Series:
    return pd.Series([0.4, 0.3, 0.2, 0.1], index=PERMNOS)


def index_series(window_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """A stand-in for sprtrn that deliberately differs from the basket, so basis is non-zero."""
    return (window_returns * weights).sum(axis=1) * 1.05


def test_window_reports_the_expected_fields(window_returns, weights):
    result = correlation_window(window_returns, index_series(window_returns, weights), weights)

    assert set(result) == {
        "n_names",
        "incomplete_in_window",
        "avg_single_vol",
        "index_vol",
        "basket_vol",
        "basis",
        "realized_correlation",
        "basket_correlation",
    }
    assert result["n_names"] == 4
    assert result["incomplete_in_window"] == 0


def test_basket_correlation_is_exact_against_the_full_matrix(window_returns, weights):
    """Zero-tolerance check. See the module docstring."""
    result = correlation_window(window_returns, index_series(window_returns, weights), weights)

    observations = window_returns.to_numpy()
    second_moments = observations.T @ observations / len(observations)
    sample_vols = np.sqrt(np.diag(second_moments))
    sample_corr = second_moments / np.outer(sample_vols, sample_vols)

    annualised = sample_vols * np.sqrt(TRADING_DAYS_PER_YEAR)
    w = weights.to_numpy()
    expected_variance = index_variance(annualised, w, sample_corr)

    assert result["basket_vol"] ** 2 == pytest.approx(expected_variance, rel=1e-12)
    assert result["basket_correlation"] == pytest.approx(
        average_correlation(np.sqrt(expected_variance), annualised, w), rel=1e-12
    )


def test_basis_is_the_difference_between_the_two_index_volatilities(window_returns, weights):
    """basis is a diagnostic for how far our basket is from the real index, so it must be exact."""
    result = correlation_window(window_returns, index_series(window_returns, weights), weights)
    assert result["basis"] == pytest.approx(result["index_vol"] - result["basket_vol"])
    assert result["index_vol"] > result["basket_vol"]  # the fixture inflates the index by 5%


def test_zero_basis_makes_the_two_correlations_agree(window_returns, weights):
    """When the index *is* the basket, the measurement and the identity must coincide."""
    basket = (window_returns * weights).sum(axis=1)
    result = correlation_window(window_returns, basket, weights)

    assert result["basis"] == pytest.approx(0.0, abs=1e-15)
    assert result["realized_correlation"] == pytest.approx(result["basket_correlation"], rel=1e-12)


def test_incomplete_names_are_dropped_counted_and_renormalised(window_returns, weights):
    """A name that halts mid-window has no usable volatility, so it leaves and the rest rescale."""
    holed = window_returns.copy()
    holed.iloc[10:, holed.columns.get_loc(10003)] = np.nan

    result = correlation_window(holed, index_series(window_returns, weights), weights)

    assert result["incomplete_in_window"] == 1
    assert result["n_names"] == 3

    # The surviving weights 0.4/0.3/0.1 renormalise to 0.5/0.375/0.125, so the weighted average
    # volatility must be computed on those rather than on the originals.
    survivors = [10001, 10002, 10004]
    renormalised = weights[survivors] / weights[survivors].sum()
    vols = np.sqrt((window_returns[survivors] ** 2).mean() * TRADING_DAYS_PER_YEAR)
    assert result["avg_single_vol"] == pytest.approx(float(renormalised @ vols))


def test_a_name_absent_from_the_window_entirely_raises(window_returns, weights):
    """The survivorship bug from the smoke test: weights for a name never fetched at all.

    This must fail loudly rather than quietly drop the name, because it means the caller fetched
    the members as of one date instead of the union over the period.
    """
    extended = pd.concat([weights, pd.Series([0.05], index=[99999])])
    extended = extended / extended.sum()

    with pytest.raises(ValueError, match="absent from window_returns entirely"):
        correlation_window(window_returns, index_series(window_returns, weights), extended)


def test_mismatched_dates_are_rejected(window_returns, weights):
    shifted = index_series(window_returns, weights)
    shifted.index = shifted.index + pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="exactly the same dates"):
        correlation_window(window_returns, shifted, weights)


def test_fewer_than_two_usable_names_raises(window_returns, weights):
    holed = window_returns.copy()
    for permno in [10002, 10003, 10004]:
        holed.iloc[5:, holed.columns.get_loc(permno)] = np.nan

    with pytest.raises(ValueError, match="correlation needs"):
        correlation_window(holed, index_series(window_returns, weights), weights)


def test_weights_must_be_a_series(window_returns):
    with pytest.raises(TypeError, match="must be a Series"):
        correlation_window(window_returns, window_returns.iloc[:, 0], np.array([0.25] * 4))


def test_measured_correlation_is_in_a_sane_range(window_returns, weights):
    """Drawn from a 0.4 correlation structure; a 21-day window is noisy but must stay plausible."""
    result = correlation_window(window_returns, index_series(window_returns, weights), weights)
    assert -0.5 < result["basket_correlation"] <= 1.0


def test_a_hole_in_the_index_series_is_rejected(window_returns, weights):
    """The index leg has no `usable` filter protecting it, so it needs an explicit check.

    `_annualised_vol` uses `.mean()`, which skips NaN silently. Without this guard a single
    missing index observation would annualise the index over 20 days while the constituents use
    21, biasing the correlation with nothing raised.
    """
    holed_index = index_series(window_returns, weights).copy()
    holed_index.iloc[7] = np.nan

    with pytest.raises(ValueError, match="missing observation"):
        correlation_window(window_returns, holed_index, weights)


def test_expected_shortages_and_pipeline_defects_raise_different_types(window_returns, weights):
    """The driver skips the first kind and must propagate the second. If these ever collapse into
    one type, `except InsufficientData` in the driver starts swallowing survivorship bugs again.
    """
    # Expected shortage: not enough names survive the window.
    holed = window_returns.copy()
    for permno in [10002, 10003, 10004]:
        holed.iloc[5:, holed.columns.get_loc(permno)] = np.nan
    with pytest.raises(InsufficientData):
        correlation_window(holed, index_series(window_returns, weights), weights)

    # Pipeline defect: a weighted name was never fetched. Must NOT be InsufficientData.
    extended = pd.concat([weights, pd.Series([0.05], index=[99999])])
    extended = extended / extended.sum()
    with pytest.raises(ValueError) as caught:
        correlation_window(window_returns, index_series(window_returns, weights), extended)
    assert not isinstance(caught.value, InsufficientData)

    # So is a hole in the index series.
    holed_index = index_series(window_returns, weights).copy()
    holed_index.iloc[0] = np.nan
    with pytest.raises(ValueError) as caught:
        correlation_window(window_returns, holed_index, weights)
    assert not isinstance(caught.value, InsufficientData)

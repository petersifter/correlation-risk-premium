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
    return (window_returns * weights).sum(axis=1) * 1.05


def test_output_fields(window_returns, weights):
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


def test_basket_correlation_exact(window_returns, weights):
    result = correlation_window(window_returns, index_series(window_returns, weights), weights)

    observations = window_returns.to_numpy()
    second_moments = observations.T @ observations / len(observations)
    sample_vols = np.sqrt(np.diag(second_moments))
    sample_corr = second_moments / np.outer(sample_vols, sample_vols)

    annualized = sample_vols * np.sqrt(TRADING_DAYS_PER_YEAR)
    w = weights.to_numpy()
    expected_variance = index_variance(annualized, w, sample_corr)

    assert result["basket_vol"] ** 2 == pytest.approx(expected_variance, rel=1e-12)
    assert result["basket_correlation"] == pytest.approx(
        average_correlation(np.sqrt(expected_variance), annualized, w), rel=1e-12
    )


def test_basis(window_returns, weights):
    result = correlation_window(window_returns, index_series(window_returns, weights), weights)
    assert result["basis"] == pytest.approx(result["index_vol"] - result["basket_vol"])
    assert result["index_vol"] > result["basket_vol"]


def test_zero_basis(window_returns, weights):
    basket = (window_returns * weights).sum(axis=1)
    result = correlation_window(window_returns, basket, weights)

    assert result["basis"] == pytest.approx(0.0, abs=1e-15)
    assert result["realized_correlation"] == pytest.approx(result["basket_correlation"], rel=1e-12)


def test_incomplete_name_dropped(window_returns, weights):
    holed = window_returns.copy()
    holed.iloc[10:, holed.columns.get_loc(10003)] = np.nan

    result = correlation_window(holed, index_series(window_returns, weights), weights)

    assert result["incomplete_in_window"] == 1
    assert result["n_names"] == 3

    survivors = [10001, 10002, 10004]
    renormalized = weights[survivors] / weights[survivors].sum()
    vols = np.sqrt((window_returns[survivors] ** 2).mean() * TRADING_DAYS_PER_YEAR)
    assert result["avg_single_vol"] == pytest.approx(float(renormalized @ vols))


def test_unfetched_name_raises(window_returns, weights):
    extended = pd.concat([weights, pd.Series([0.05], index=[99999])])
    extended = extended / extended.sum()

    with pytest.raises(ValueError, match="absent from window_returns entirely"):
        correlation_window(window_returns, index_series(window_returns, weights), extended)


def test_mismatched_dates(window_returns, weights):
    shifted = index_series(window_returns, weights)
    shifted.index = shifted.index + pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="exactly the same dates"):
        correlation_window(window_returns, shifted, weights)


def test_fewer_than_two_names(window_returns, weights):
    holed = window_returns.copy()
    for permno in [10002, 10003, 10004]:
        holed.iloc[5:, holed.columns.get_loc(permno)] = np.nan

    with pytest.raises(ValueError, match="correlation needs"):
        correlation_window(holed, index_series(window_returns, weights), weights)


def test_weights_must_be_series(window_returns):
    with pytest.raises(TypeError, match="must be a Series"):
        correlation_window(window_returns, window_returns.iloc[:, 0], np.array([0.25] * 4))


def test_correlation_plausible(window_returns, weights):
    result = correlation_window(window_returns, index_series(window_returns, weights), weights)
    assert -0.5 < result["basket_correlation"] <= 1.0


def test_index_gap_rejected(window_returns, weights):
    holed_index = index_series(window_returns, weights).copy()
    holed_index.iloc[7] = np.nan

    with pytest.raises(ValueError, match="missing observation"):
        correlation_window(window_returns, holed_index, weights)


def test_shortage_vs_defect_exception_types(window_returns, weights):
    holed = window_returns.copy()
    for permno in [10002, 10003, 10004]:
        holed.iloc[5:, holed.columns.get_loc(permno)] = np.nan
    with pytest.raises(InsufficientData):
        correlation_window(holed, index_series(window_returns, weights), weights)

    extended = pd.concat([weights, pd.Series([0.05], index=[99999])])
    extended = extended / extended.sum()
    with pytest.raises(ValueError) as caught:
        correlation_window(window_returns, index_series(window_returns, weights), extended)
    assert not isinstance(caught.value, InsufficientData)

    holed_index = index_series(window_returns, weights).copy()
    holed_index.iloc[0] = np.nan
    with pytest.raises(ValueError) as caught:
        correlation_window(window_returns, holed_index, weights)
    assert not isinstance(caught.value, InsufficientData)

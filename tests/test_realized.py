from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.correlation import average_correlation, index_variance
from dispersion.realized import TRADING_DAYS_PER_YEAR, annualized_vol, basket_return

WINDOW = 21


def synthetic_returns(
    n_days: int = 400,
    n_names: int = 6,
    true_correlation: float = 0.35,
    daily_vol: float = 0.02,
    seed: int = 12345,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    corr = np.full((n_names, n_names), true_correlation)
    np.fill_diagonal(corr, 1.0)

    draws = rng.multivariate_normal(np.zeros(n_names), corr * daily_vol**2, size=n_days)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    names = [f"NAME{i}" for i in range(n_names)]
    return pd.DataFrame(draws, index=dates, columns=names)


def equal_weights(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns)


def test_constant_magnitude_returns():
    returns = pd.Series([0.01, -0.01] * 50)
    assert annualized_vol(returns) == pytest.approx(0.01 * np.sqrt(TRADING_DAYS_PER_YEAR))


def test_series_and_frame_outputs():
    returns = synthetic_returns(n_days=60, n_names=4)

    assert isinstance(annualized_vol(returns["NAME0"]), float)
    per_column = annualized_vol(returns)
    assert isinstance(per_column, pd.Series)
    assert list(per_column.index) == list(returns.columns)


def test_demean_option():
    returns = synthetic_returns(n_days=100, n_names=2)["NAME0"]

    undemeaned = annualized_vol(returns)
    demeaned = annualized_vol(returns, demean=True)

    assert demeaned == pytest.approx(float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    assert undemeaned != pytest.approx(demeaned)


def test_sqrt_time_scaling():
    returns = synthetic_returns(n_days=60, n_names=2)["NAME0"]
    assert annualized_vol(returns, trading_days=4 * TRADING_DAYS_PER_YEAR) == pytest.approx(
        2 * annualized_vol(returns)
    )


def test_nan_rejected():
    returns = synthetic_returns(n_days=30)
    returns.iloc[5, 2] = np.nan

    with pytest.raises(ValueError, match="contains NaN"):
        annualized_vol(returns)


def test_basket_return_weighted_sum():
    returns = synthetic_returns(n_days=10, n_names=3)
    weights = pd.Series([0.2, 0.3, 0.5], index=returns.columns)

    expected = 0.2 * returns["NAME0"] + 0.3 * returns["NAME1"] + 0.5 * returns["NAME2"]
    pd.testing.assert_series_equal(basket_return(returns, weights), expected)


def test_basket_identity_exact():
    returns = synthetic_returns().iloc[-WINDOW:]
    weights = equal_weights(returns)

    component_vols = annualized_vol(returns)
    index_vol = annualized_vol(basket_return(returns, weights))
    rho_bar = average_correlation(index_vol, component_vols.to_numpy(), weights.to_numpy())

    observations = returns.to_numpy()
    second_moments = observations.T @ observations / len(observations)
    sample_vols = np.sqrt(np.diag(second_moments))
    sample_corr = second_moments / np.outer(sample_vols, sample_vols)

    annualized = sample_vols * np.sqrt(TRADING_DAYS_PER_YEAR)
    w = weights.to_numpy()
    expected_variance = index_variance(annualized, w, sample_corr)

    assert index_vol**2 == pytest.approx(expected_variance, rel=1e-12)
    assert rho_bar == pytest.approx(
        average_correlation(np.sqrt(expected_variance), annualized, w), rel=1e-12
    )


def test_recovers_true_correlation():
    returns = synthetic_returns(n_days=3000, true_correlation=0.35)
    weights = equal_weights(returns)

    rho_bar = average_correlation(
        annualized_vol(basket_return(returns, weights)),
        annualized_vol(returns).to_numpy(),
        weights.to_numpy(),
    )
    assert rho_bar == pytest.approx(0.35, abs=0.02)


def test_basket_weights_sum():
    returns = synthetic_returns(n_days=10, n_names=3)
    weights = pd.Series([0.2, 0.3, 0.4], index=returns.columns)

    with pytest.raises(ValueError, match="must sum to 1"):
        basket_return(returns, weights)


def test_basket_weights_align():
    returns = synthetic_returns(n_days=10, n_names=3)
    weights = pd.Series([0.2, 0.3, 0.5], index=["NAME0", "NAME1", "OTHER"])

    with pytest.raises(ValueError, match="must match exactly"):
        basket_return(returns, weights)


def test_basket_weights_type():
    returns = synthetic_returns(n_days=10, n_names=3)

    with pytest.raises(TypeError, match="must be a Series"):
        basket_return(returns, np.array([0.2, 0.3, 0.5]))


def test_single_name_basket():
    returns = synthetic_returns(n_days=10, n_names=2)[["NAME0"]]
    weights = pd.Series([1.0], index=returns.columns)

    with pytest.raises(ValueError, match="at least two"):
        basket_return(returns, weights)

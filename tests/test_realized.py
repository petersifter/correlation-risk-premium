"""Tests for the shared volatility primitives.

These two functions are the only arithmetic used by both the realised and the implied measurement,
so the properties asserted here are the ones every number downstream inherits.

The load-bearing one is ``test_constructed_basket_satisfies_the_identity_exactly``. Because
volatilities are not demeaned, the matrix of sample second moments *is* a covariance matrix, so a
basket reconstructed from its own constituents satisfies rung 1's identity to floating-point
precision. If that ever fails, the pipeline is broken rather than noisy.
"""

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
    """A panel of daily returns drawn from a known homogeneous correlation structure."""
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
    """Returns of +/-1% every day have realised volatility exactly 0.01 * sqrt(252)."""
    returns = pd.Series([0.01, -0.01] * 50)
    assert annualized_vol(returns) == pytest.approx(0.01 * np.sqrt(TRADING_DAYS_PER_YEAR))


def test_a_series_returns_a_scalar_and_a_frame_returns_one_per_column():
    returns = synthetic_returns(n_days=60, n_names=4)

    assert isinstance(annualized_vol(returns["NAME0"]), float)
    per_column = annualized_vol(returns)
    assert isinstance(per_column, pd.Series)
    assert list(per_column.index) == list(returns.columns)


def test_demeaning_changes_the_estimate_and_matches_pandas():
    """demean=True is the sample standard deviation; the default is the raw second moment."""
    returns = synthetic_returns(n_days=100, n_names=2)["NAME0"]

    undemeaned = annualized_vol(returns)
    demeaned = annualized_vol(returns, demean=True)

    assert demeaned == pytest.approx(float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR))
    assert undemeaned != pytest.approx(demeaned)


def test_annualisation_scales_with_the_square_root_of_time():
    returns = synthetic_returns(n_days=60, n_names=2)["NAME0"]
    assert annualized_vol(returns, trading_days=4 * TRADING_DAYS_PER_YEAR) == pytest.approx(
        2 * annualized_vol(returns)
    )


def test_returns_containing_nan_are_rejected():
    """mean() skips NaN silently, so an unchecked hole would annualise over fewer observations."""
    returns = synthetic_returns(n_days=30)
    returns.iloc[5, 2] = np.nan

    with pytest.raises(ValueError, match="contains NaN"):
        annualized_vol(returns)


def test_basket_return_is_the_weighted_sum():
    returns = synthetic_returns(n_days=10, n_names=3)
    weights = pd.Series([0.2, 0.3, 0.5], index=returns.columns)

    expected = 0.2 * returns["NAME0"] + 0.3 * returns["NAME1"] + 0.5 * returns["NAME2"]
    pd.testing.assert_series_equal(basket_return(returns, weights), expected)


def test_constructed_basket_satisfies_the_identity_exactly():
    """The load-bearing test. See the module docstring.

    Build the index from its constituents, measure non-demeaned volatilities, and the average
    correlation that comes back must equal the one implied by the full sample correlation matrix -
    to floating-point precision, with no tolerance for sampling error.
    """
    returns = synthetic_returns().iloc[-WINDOW:]
    weights = equal_weights(returns)

    component_vols = annualized_vol(returns)
    index_vol = annualized_vol(basket_return(returns, weights))
    rho_bar = average_correlation(index_vol, component_vols.to_numpy(), weights.to_numpy())

    observations = returns.to_numpy()
    second_moments = observations.T @ observations / len(observations)
    sample_vols = np.sqrt(np.diag(second_moments))
    sample_corr = second_moments / np.outer(sample_vols, sample_vols)

    annualised = sample_vols * np.sqrt(TRADING_DAYS_PER_YEAR)
    w = weights.to_numpy()
    expected_variance = index_variance(annualised, w, sample_corr)

    assert index_vol**2 == pytest.approx(expected_variance, rel=1e-12)
    assert rho_bar == pytest.approx(
        average_correlation(np.sqrt(expected_variance), annualised, w), rel=1e-12
    )


def test_measured_correlation_recovers_the_true_correlation():
    """Sanity, not precision: a long sample on a known structure should land near the truth."""
    returns = synthetic_returns(n_days=3000, true_correlation=0.35)
    weights = equal_weights(returns)

    rho_bar = average_correlation(
        annualized_vol(basket_return(returns, weights)),
        annualized_vol(returns).to_numpy(),
        weights.to_numpy(),
    )
    assert rho_bar == pytest.approx(0.35, abs=0.02)


def test_basket_weights_must_sum_to_one():
    returns = synthetic_returns(n_days=10, n_names=3)
    weights = pd.Series([0.2, 0.3, 0.4], index=returns.columns)

    with pytest.raises(ValueError, match="must sum to 1"):
        basket_return(returns, weights)


def test_basket_weights_must_align_with_the_columns():
    returns = synthetic_returns(n_days=10, n_names=3)
    weights = pd.Series([0.2, 0.3, 0.5], index=["NAME0", "NAME1", "OTHER"])

    with pytest.raises(ValueError, match="must match exactly"):
        basket_return(returns, weights)


def test_basket_weights_must_be_a_series():
    returns = synthetic_returns(n_days=10, n_names=3)

    with pytest.raises(TypeError, match="must be a Series"):
        basket_return(returns, np.array([0.2, 0.3, 0.5]))


def test_single_constituent_basket_is_rejected():
    returns = synthetic_returns(n_days=10, n_names=2)[["NAME0"]]
    weights = pd.Series([1.0], index=returns.columns)

    with pytest.raises(ValueError, match="at least two"):
        basket_return(returns, weights)

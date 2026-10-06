"""Tests for rung 2: realised volatility and realised correlation.

Two of these tests are the load-bearing ones.

``test_constructed_basket_satisfies_the_identity_exactly`` is the proof that the pipeline is
correct. When the index return is built from the constituents and volatilities are not demeaned,
rung 1's identity holds *exactly* on sample moments - not approximately, not up to sampling error.
Writing out ``mean(R_I^2) = sum_i sum_j w_i w_j mean(R_i R_j)`` and dividing through by the sample
volatilities shows why: the non-demeaned sample second moments *are* a covariance matrix. If this
test ever fails, the data pipeline is broken, not noisy.

``test_panel_matches_the_scalar_reference_row_by_row`` keeps the vectorised implementation honest
against the simple scalar one from rung 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dispersion.correlation import average_correlation, index_variance
from dispersion.realized import (
    TRADING_DAYS_PER_YEAR,
    average_correlation_panel,
    basket_return,
    realized_vol,
)

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
    covariance = corr * daily_vol**2

    draws = rng.multivariate_normal(np.zeros(n_names), covariance, size=n_days)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    names = [f"NAME{i}" for i in range(n_names)]
    return pd.DataFrame(draws, index=dates, columns=names)


def equal_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """Equal weights aligned to a return panel."""
    n = returns.shape[1]
    return pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)


def test_realized_vol_of_constant_magnitude_returns():
    """Returns of +/-1% every day have realised volatility exactly 0.01 * sqrt(252)."""
    returns = pd.Series([0.01, -0.01] * 50, index=pd.bdate_range("2020-01-01", periods=100))
    vol = realized_vol(returns, window=WINDOW)

    expected = 0.01 * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert vol.dropna().to_numpy() == pytest.approx(expected)


def test_realized_vol_window_is_trailing_and_right_closed():
    """sigma_t uses returns up to and including t, so the first window-1 rows are NaN."""
    returns = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=50))
    vol = realized_vol(returns, window=WINDOW)

    assert vol.iloc[: WINDOW - 1].isna().all()
    assert vol.iloc[WINDOW - 1 :].notna().all()


def test_demeaning_changes_the_estimate_and_matches_pandas():
    """demean=True is the sample standard deviation; the default is the raw second moment."""
    returns = synthetic_returns(n_days=100, n_names=2)["NAME0"]

    undemeaned = realized_vol(returns, window=WINDOW)
    demeaned = realized_vol(returns, window=WINDOW, demean=True)

    expected = returns.rolling(WINDOW).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    pd.testing.assert_series_equal(demeaned, expected)
    assert not np.allclose(undemeaned.dropna(), demeaned.dropna())


def test_realized_vol_rejects_a_degenerate_window():
    with pytest.raises(ValueError, match="at least 2"):
        realized_vol(pd.Series([0.01, 0.02]), window=1)


def test_basket_return_is_the_weighted_sum():
    returns = synthetic_returns(n_days=10, n_names=3)
    weights = pd.DataFrame([[0.2, 0.3, 0.5]] * 10, index=returns.index, columns=returns.columns)

    basket = basket_return(returns, weights)
    expected = 0.2 * returns["NAME0"] + 0.3 * returns["NAME1"] + 0.5 * returns["NAME2"]
    pd.testing.assert_series_equal(basket, expected)


def test_constructed_basket_satisfies_the_identity_exactly():
    """The load-bearing test. See the module docstring.

    Build the index from its constituents, measure non-demeaned realised volatilities, and the
    average correlation that comes back must equal the one implied by the full sample correlation
    matrix - to floating-point precision, with no tolerance for sampling error.
    """
    returns = synthetic_returns()
    weights = equal_weights(returns)
    w = weights.iloc[0].to_numpy()

    index_returns = basket_return(returns, weights)
    component_vols = realized_vol(returns, window=WINDOW)
    index_vol = realized_vol(index_returns, window=WINDOW)

    rho_bar = average_correlation_panel(index_vol, component_vols, weights)

    # Independently reconstruct the final window from its sample second moments. Because the
    # volatilities are not demeaned, the matrix of second moments *is* the covariance matrix.
    final_window = returns.iloc[-WINDOW:].to_numpy()
    second_moments = final_window.T @ final_window / WINDOW
    sample_vols = np.sqrt(np.diag(second_moments))
    sample_corr = second_moments / np.outer(sample_vols, sample_vols)

    annualised_vols = sample_vols * np.sqrt(TRADING_DAYS_PER_YEAR)
    reconstructed_variance = index_variance(annualised_vols, w, sample_corr)
    reconstructed_rho = average_correlation(np.sqrt(reconstructed_variance), annualised_vols, w)

    assert rho_bar.iloc[-1] == pytest.approx(reconstructed_rho, rel=1e-12)
    assert index_vol.iloc[-1] ** 2 == pytest.approx(reconstructed_variance, rel=1e-12)


def test_measured_correlation_recovers_the_true_correlation():
    """Sanity, not precision: a long window on a known structure should land near the truth."""
    returns = synthetic_returns(n_days=3000, true_correlation=0.35)
    weights = equal_weights(returns)

    index_returns = basket_return(returns, weights)
    rho_bar = average_correlation_panel(
        realized_vol(index_returns, window=252),
        realized_vol(returns, window=252),
        weights,
    )
    assert rho_bar.dropna().mean() == pytest.approx(0.35, abs=0.03)


def test_panel_matches_the_scalar_reference_row_by_row():
    """The vectorised panel must agree with rung 1's scalar function on every row."""
    returns = synthetic_returns(n_days=120)
    weights = equal_weights(returns)
    w = weights.iloc[0].to_numpy()

    index_returns = basket_return(returns, weights)
    component_vols = realized_vol(returns, window=WINDOW)
    index_vol = realized_vol(index_returns, window=WINDOW)

    panel = average_correlation_panel(index_vol, component_vols, weights)

    complete = component_vols.dropna().index
    scalar = pd.Series(
        [
            average_correlation(index_vol[date], component_vols.loc[date].to_numpy(), w)
            for date in complete
        ],
        index=complete,
    )
    pd.testing.assert_series_equal(panel.loc[complete], scalar, check_names=False)


def test_leading_nan_rows_propagate_rather_than_raise():
    """The first window-1 rows of a rolling volatility are legitimately NaN, not a data error."""
    returns = synthetic_returns(n_days=60)
    weights = equal_weights(returns)

    index_returns = basket_return(returns, weights)
    rho_bar = average_correlation_panel(
        realized_vol(index_returns, window=WINDOW),
        realized_vol(returns, window=WINDOW),
        weights,
    )
    assert rho_bar.iloc[: WINDOW - 1].isna().all()
    assert rho_bar.iloc[WINDOW - 1 :].notna().all()


def test_returns_containing_nan_are_rejected():
    returns = synthetic_returns(n_days=30)
    returns.iloc[5, 2] = np.nan

    with pytest.raises(ValueError, match="resolve missing observations at the data layer"):
        realized_vol(returns, window=WINDOW)


def test_weights_must_sum_to_one_on_every_row():
    returns = synthetic_returns(n_days=30)
    weights = equal_weights(returns)
    weights.iloc[10, 0] = 0.9

    with pytest.raises(ValueError, match="must sum to 1"):
        basket_return(returns, weights)


def test_misaligned_columns_are_rejected():
    returns = synthetic_returns(n_days=30)
    weights = equal_weights(returns).rename(columns={"NAME0": "OTHER"})

    with pytest.raises(ValueError, match="same columns"):
        basket_return(returns, weights)


def test_misaligned_index_is_rejected():
    returns = synthetic_returns(n_days=30)
    weights = equal_weights(returns)
    weights.index = pd.bdate_range("2001-01-01", periods=30)

    with pytest.raises(ValueError, match="same index"):
        basket_return(returns, weights)


def test_single_constituent_is_rejected():
    returns = synthetic_returns(n_days=30, n_names=2)[["NAME0"]]
    weights = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)

    with pytest.raises(ValueError, match="at least two"):
        basket_return(returns, weights)

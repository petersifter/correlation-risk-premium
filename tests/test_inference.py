"""Tests for rung 4's inference.

The two that matter are ``test_newey_west_matches_statsmodels`` and
``test_hac_regression_matches_statsmodels``. Newey-West is implemented by hand here because the
arithmetic is short and worth owning, but hand-rolled covariance estimators are exactly the kind of
code that is subtly wrong and still looks plausible. Checking against ``statsmodels`` on the same
lag length removes that risk without giving up the ownership.

The rest assert the properties that make the estimator worth using at all: that it widens under
positive serial correlation, that it reduces to the naive standard error when there is none, and
that a known regression is recovered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from dispersion.inference import (
    bartlett_lag_default,
    forecast_regression,
    newey_west_mean,
    subsample_means,
)


def ar1(n: int, rho: float, seed: int = 11, scale: float = 1.0, mean: float = 0.0) -> pd.Series:
    """An AR(1) series with known persistence, as a stand-in for a correlation premium."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(scale=scale, size=n)
    values = np.empty(n)
    values[0] = shocks[0]
    for t in range(1, n):
        values[t] = rho * values[t - 1] + shocks[t]
    return pd.Series(values + mean, index=pd.bdate_range("2000-01-01", periods=n, freq="ME"))


def test_newey_west_matches_statsmodels():
    """The long-run variance must agree with a reference implementation on the same lag."""
    series = ar1(300, rho=0.6)
    lags = bartlett_lag_default(len(series))

    ours = newey_west_mean(series, lags=lags)

    design = np.ones((len(series), 1))
    reference = sm.OLS(series.to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )

    assert ours.mean == pytest.approx(float(reference.params[0]))
    assert ours.std_error == pytest.approx(float(reference.bse[0]), rel=1e-10)
    assert ours.t_statistic == pytest.approx(float(reference.tvalues[0]), rel=1e-10)


def test_hac_regression_matches_statsmodels():
    """Coefficients and HAC standard errors must both agree with the reference."""
    rng = np.random.default_rng(5)
    implied = ar1(400, rho=0.7, mean=0.4)
    realized = 0.5 * implied + pd.Series(
        rng.normal(scale=0.2, size=len(implied)), index=implied.index
    )

    lags = bartlett_lag_default(len(implied))
    ours = forecast_regression(realized, implied, lags=lags)

    design = sm.add_constant(implied.to_numpy())
    reference = sm.OLS(realized.to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )

    assert ours.alpha == pytest.approx(float(reference.params[0]), rel=1e-10)
    assert ours.beta == pytest.approx(float(reference.params[1]), rel=1e-10)
    assert ours.alpha_std_error == pytest.approx(float(reference.bse[0]), rel=1e-10)
    assert ours.beta_std_error == pytest.approx(float(reference.bse[1]), rel=1e-10)
    assert ours.r_squared == pytest.approx(float(reference.rsquared), rel=1e-10)


def test_serial_correlation_widens_the_standard_error():
    """The entire reason the estimator exists: a naive standard error is too small when rho > 0."""
    persistent = newey_west_mean(ar1(400, rho=0.8))
    assert persistent.inflation > 1.5


def test_white_noise_leaves_the_standard_error_roughly_alone():
    """With no serial correlation the robust and naive estimates should be close."""
    independent = newey_west_mean(ar1(400, rho=0.0))
    assert independent.inflation == pytest.approx(1.0, abs=0.15)


def test_zero_lags_reduces_to_the_sample_variance():
    """With lags=0 the long-run variance is just gamma_0, so only the ddof differs from naive."""
    series = ar1(200, rho=0.3)
    estimate = newey_west_mean(series, lags=0)

    expected = float(np.sqrt(series.var(ddof=0) / len(series)))
    assert estimate.std_error == pytest.approx(expected)


def test_the_automatic_lag_rule():
    """Newey-West (1994): floor(4 (T/100)^(2/9)), floored at 1."""
    assert bartlett_lag_default(100) == 4
    assert bartlett_lag_default(346) == 5
    assert bartlett_lag_default(2) == 1


def test_a_known_regression_is_recovered():
    """A clean linear relationship must come back with the coefficients that generated it."""
    implied = pd.Series(np.linspace(0.1, 0.9, 200))
    realized = 0.05 + 0.8 * implied

    fit = forecast_regression(realized, implied, lags=2)
    assert fit.alpha == pytest.approx(0.05)
    assert fit.beta == pytest.approx(0.8)
    assert fit.r_squared == pytest.approx(1.0)


def test_t_beta_equals_one_tests_against_one_not_zero():
    """An unbiased forecast has beta = 1, so that is the null the statistic must use.

    Tested on a noisy half-slope rather than a perfect fit: with zero residuals both the numerator
    and the standard error collapse to floating-point dust and their ratio is meaningless.
    """
    rng = np.random.default_rng(3)
    implied = pd.Series(np.linspace(0.1, 0.9, 300))
    realized = 0.5 * implied + pd.Series(rng.normal(scale=0.05, size=300))

    fit = forecast_regression(realized, implied, lags=3)

    assert fit.beta == pytest.approx(0.5, abs=0.05)
    assert fit.t_beta_equals_one == pytest.approx((fit.beta - 1.0) / fit.beta_std_error, rel=1e-12)
    # A slope near 0.5 is emphatically not 1, so the null must be rejected hard.
    assert fit.t_beta_equals_one < -5


def test_regression_aligns_and_drops_unpaired_rows():
    implied = ar1(100, rho=0.4)
    realized = implied * 0.7
    realized.iloc[:10] = np.nan

    fit = forecast_regression(realized, implied)
    assert fit.n_obs == 90


def test_subsample_means_carry_their_own_error_bars():
    """A table of point estimates invites a trend; each group needs its own interval."""
    # Noise scale well below the 0.12 gap between the group means, or the ordering is chance.
    series = pd.concat(
        [
            ar1(120, rho=0.5, mean=0.15, scale=0.02, seed=1),
            ar1(120, rho=0.5, mean=0.03, scale=0.02, seed=2),
        ]
    )
    series.index = pd.RangeIndex(len(series))
    groups = pd.Series(["early"] * 120 + ["late"] * 120, index=series.index)

    table = subsample_means(series, groups)
    assert list(table.index) == ["early", "late"]
    assert (table["ci_low"] < table["mean"]).all()
    assert (table["mean"] < table["ci_high"]).all()
    assert table.loc["early", "mean"] > table.loc["late", "mean"]


def test_too_few_observations_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        newey_west_mean(pd.Series([0.1]))
    with pytest.raises(ValueError, match="at least three"):
        forecast_regression(pd.Series([0.1, 0.2]), pd.Series([0.3, 0.4]))


def test_lag_longer_than_the_sample_is_rejected():
    with pytest.raises(ValueError, match="lags must be in"):
        newey_west_mean(ar1(20, rho=0.2), lags=20)

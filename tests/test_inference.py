from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from dispersion.inference import (
    bartlett_lag_default,
    block_bootstrap_mean,
    forecast_regression,
    newey_west_mean,
    subsample_means,
)


def ar1(n: int, rho: float, seed: int = 11, scale: float = 1.0, mean: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(scale=scale, size=n)
    values = np.empty(n)
    values[0] = shocks[0]
    for t in range(1, n):
        values[t] = rho * values[t - 1] + shocks[t]
    return pd.Series(values + mean, index=pd.bdate_range("2000-01-01", periods=n, freq="ME"))


def test_newey_west_matches_statsmodels():
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


def test_persistence_widens_se():
    persistent = newey_west_mean(ar1(400, rho=0.8))
    assert persistent.inflation > 1.5


def test_white_noise_se_close_to_naive():
    independent = newey_west_mean(ar1(400, rho=0.0))
    assert independent.inflation == pytest.approx(1.0, abs=0.15)


def test_zero_lags_is_sample_variance():
    series = ar1(200, rho=0.3)
    estimate = newey_west_mean(series, lags=0)

    expected = float(np.sqrt(series.var(ddof=0) / len(series)))
    assert estimate.std_error == pytest.approx(expected)


def test_automatic_lag():
    assert bartlett_lag_default(100) == 4
    assert bartlett_lag_default(346) == 5
    assert bartlett_lag_default(2) == 1


def test_known_regression():
    implied = pd.Series(np.linspace(0.1, 0.9, 200))
    realized = 0.05 + 0.8 * implied

    fit = forecast_regression(realized, implied, lags=2)
    assert fit.alpha == pytest.approx(0.05)
    assert fit.beta == pytest.approx(0.8)
    assert fit.r_squared == pytest.approx(1.0)


def test_beta_t_stat_against_one():
    rng = np.random.default_rng(3)
    implied = pd.Series(np.linspace(0.1, 0.9, 300))
    realized = 0.5 * implied + pd.Series(rng.normal(scale=0.05, size=300))

    fit = forecast_regression(realized, implied, lags=3)

    assert fit.beta == pytest.approx(0.5, abs=0.05)
    assert fit.t_beta_equals_one == pytest.approx((fit.beta - 1.0) / fit.beta_std_error, rel=1e-12)
    assert fit.t_beta_equals_one < -5


def test_regression_drops_unpaired_rows():
    implied = ar1(100, rho=0.4)
    realized = implied * 0.7
    realized.iloc[:10] = np.nan

    fit = forecast_regression(realized, implied)
    assert fit.n_obs == 90


def test_subsample_means():
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


def test_block_bootstrap_brackets_the_mean():
    series = ar1(300, rho=0.5, mean=1.0)
    low, high = block_bootstrap_mean(series, draws=2_000)
    assert low < series.mean() < high


def test_block_bootstrap_wider_than_iid():
    series = ar1(400, rho=0.8)
    iid_low, iid_high = block_bootstrap_mean(series, block=1, draws=4_000)
    blk_low, blk_high = block_bootstrap_mean(series, block=12, draws=4_000)
    assert blk_high - blk_low > 1.5 * (iid_high - iid_low)


def test_block_bootstrap_is_seeded():
    series = ar1(100, rho=0.2)
    assert block_bootstrap_mean(series, seed=3) == block_bootstrap_mean(series, seed=3)


def test_too_few_observations():
    with pytest.raises(ValueError, match="at least two"):
        newey_west_mean(pd.Series([0.1]))
    with pytest.raises(ValueError, match="at least three"):
        forecast_regression(pd.Series([0.1, 0.2]), pd.Series([0.3, 0.4]))


def test_lag_too_long():
    with pytest.raises(ValueError, match="lags must be in"):
        newey_west_mean(ar1(20, rho=0.2), lags=20)

"""Rung 4: inference on the correlation risk premium.

Rung 3 produced a mean premium of about +0.072. This module decides whether that number is worth
anything, which comes down to three questions.

Why a naive standard error is not good enough
---------------------------------------------
Correlation is persistent: a high-correlation month tends to follow a high-correlation month, and
the premium inherits that persistence. The usual standard error of a mean, ``s / sqrt(T)``, assumes
independent observations. Under positive serial correlation the effective number of independent
observations is smaller than ``T``, so that formula understates the uncertainty and the t-statistic
is too large - sometimes by a factor of two or more.

:func:`newey_west_mean` fixes this. The long-run variance of a serially correlated series is not
the variance ``gamma_0`` but the sum of all autocovariances::

    S = gamma_0 + 2 sum_{j=1}^{L} w_j gamma_j,    w_j = 1 - j / (L + 1)

The ``w_j`` are Bartlett weights, which taper to zero at lag ``L + 1``. They are not cosmetic: a
raw truncated sum can produce a negative variance estimate, and the Bartlett taper guarantees the
estimator stays non-negative (Newey and West, 1987). The standard error of the mean is then
``sqrt(S / T)``.

The default lag length is the Newey-West (1994) automatic rule ``L = floor(4 (T/100)^(2/9))``,
which for a few hundred monthly observations gives four or five lags.

Is implied correlation an unbiased forecast
-------------------------------------------
:func:`forecast_regression` runs the Mincer-Zarnowitz regression::

    realised_t = alpha + beta * implied_t + e_t

If the options market forecast correlation without bias, ``alpha = 0`` and ``beta = 1``. Two
distinct failures are worth separating:

* ``alpha < 0`` with ``beta = 1`` is a *level* effect - the market charges a constant premium over
  what it expects. That is a risk premium.
* ``beta < 1`` is an *over-reaction* effect - the market moves its forecast around more than
  correlation actually moves, so a high implied reading forecasts proportionally less than it
  claims. That is a different, and conditionally tradeable, statement.

Standard errors here are HAC (heteroskedasticity and autocorrelation consistent) for the same
reason as above, using the same Bartlett kernel in the sandwich form
``(X'X)^-1 S (X'X)^-1``.

A note on which measurement error matters
-----------------------------------------
Rung 2 established that a 21-day realised correlation carries a standard error of about 0.065. That
noise sits in the *dependent* variable of the regression above, where it inflates the residual
variance and lowers R-squared but leaves ``beta`` unbiased. Attenuation of ``beta`` towards zero
would require noise in the *regressor* - the implied side - which comes from surface interpolation
and stale quotes and is a smaller and separate problem. The two are easy to conflate and the
distinction changes what the regression can be asked to prove.

Everything here is implemented directly rather than called from a library, because the arithmetic
is short and an interview will ask how it works. ``tests/test_inference.py`` checks it against
``statsmodels``, so owning the implementation costs nothing in correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ForecastRegression",
    "HacRegression",
    "MeanEstimate",
    "bartlett_lag_default",
    "forecast_regression",
    "hac_regression",
    "newey_west_mean",
    "subsample_means",
]


def bartlett_lag_default(n_obs: int) -> int:
    """Newey-West (1994) automatic lag: ``floor(4 (T/100)^(2/9))``, at least 1."""
    if n_obs < 2:
        raise ValueError(f"need at least two observations, got {n_obs}")
    return max(1, int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


@dataclass(frozen=True)
class MeanEstimate:
    """A sample mean with a serial-correlation-robust standard error."""

    mean: float
    std_error: float
    t_statistic: float
    n_obs: int
    lags: int
    naive_std_error: float

    @property
    def inflation(self) -> float:
        """How much wider the robust standard error is than the naive one."""
        return self.std_error / self.naive_std_error


def _bartlett_long_run_variance(x: np.ndarray, lags: int) -> float:
    """``S = gamma_0 + 2 sum_j w_j gamma_j`` with Bartlett weights ``w_j = 1 - j/(lags+1)``."""
    n = x.size
    deviations = x - x.mean()

    long_run = float(deviations @ deviations) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = float(deviations[lag:] @ deviations[:-lag]) / n
        long_run += 2.0 * weight * autocovariance
    return long_run


def newey_west_mean(series: pd.Series, lags: int | None = None) -> MeanEstimate:
    """Mean of ``series`` with a Newey-West standard error.

    ``lags`` defaults to :func:`bartlett_lag_default`. The returned
    :attr:`MeanEstimate.inflation` says how much the serial correlation widened the interval, which
    is usually the number worth quoting: it is the factor by which a naive t-statistic was wrong.
    """
    values = series.dropna().to_numpy(dtype=float)
    n = values.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}")

    lags = bartlett_lag_default(n) if lags is None else lags
    if lags < 0 or lags >= n:
        raise ValueError(f"lags must be in [0, {n - 1}], got {lags}")

    long_run = _bartlett_long_run_variance(values, lags)
    if long_run <= 0.0:
        raise ValueError(
            f"long-run variance estimate is {long_run:.6g}, which is not positive; the Bartlett "
            "kernel should prevent this, so suspect a degenerate series"
        )

    mean = float(values.mean())
    std_error = float(np.sqrt(long_run / n))
    naive = float(values.std(ddof=1) / np.sqrt(n))

    return MeanEstimate(
        mean=mean,
        std_error=std_error,
        t_statistic=mean / std_error,
        n_obs=n,
        lags=lags,
        naive_std_error=naive,
    )


@dataclass(frozen=True)
class ForecastRegression:
    """A Mincer-Zarnowitz regression of realised on implied, with HAC standard errors."""

    alpha: float
    beta: float
    alpha_std_error: float
    beta_std_error: float
    r_squared: float
    n_obs: int
    lags: int

    @property
    def t_alpha(self) -> float:
        """t-statistic for ``alpha = 0``: no constant forecast bias."""
        return self.alpha / self.alpha_std_error

    @property
    def t_beta_equals_one(self) -> float:
        """t-statistic for ``beta = 1``: the market's forecast moves one-for-one with the outcome.

        Testing against 1 rather than 0 is the point. A ``beta`` reliably below 1 means implied
        correlation over-reacts, which is a conditional statement about when the premium is largest,
        not merely evidence that implied correlation is informative at all.
        """
        return (self.beta - 1.0) / self.beta_std_error


def forecast_regression(
    realized: pd.Series,
    implied: pd.Series,
    lags: int | None = None,
) -> ForecastRegression:
    """Regress ``realized`` on ``implied`` with Newey-West HAC standard errors.

    ``realised_t = alpha + beta * implied_t + e_t``. The two series are aligned on their index and
    rows missing either side are dropped.
    """
    fit = hac_regression(realized, {"implied": implied}, lags=lags)
    return ForecastRegression(
        alpha=float(fit.coefficients["const"]),
        beta=float(fit.coefficients["implied"]),
        alpha_std_error=float(fit.std_errors["const"]),
        beta_std_error=float(fit.std_errors["implied"]),
        r_squared=fit.r_squared,
        n_obs=fit.n_obs,
        lags=fit.lags,
    )


@dataclass(frozen=True)
class HacRegression:
    """An OLS fit with heteroskedasticity- and autocorrelation-consistent standard errors."""

    coefficients: pd.Series
    std_errors: pd.Series
    r_squared: float
    n_obs: int
    lags: int

    @property
    def t_statistics(self) -> pd.Series:
        """t-statistics against a null of zero."""
        return (self.coefficients / self.std_errors).rename("t")

    def summary(self) -> pd.DataFrame:
        """One row per regressor: estimate, standard error, t."""
        return pd.DataFrame(
            {
                "estimate": self.coefficients,
                "std_error": self.std_errors,
                "t": self.t_statistics,
            }
        )


def hac_regression(
    y: pd.Series,
    regressors: dict[str, pd.Series],
    lags: int | None = None,
) -> HacRegression:
    """OLS of ``y`` on ``regressors`` with a constant, using Newey-West HAC standard errors.

    The single implementation of the HAC sandwich in this package. ``forecast_regression`` is a
    named wrapper around it, and rung 6's exposure analysis uses it directly, so there is one place
    where the covariance estimator lives and no chance of two copies drifting apart.

    The covariance is ``(X'X)^-1 S (X'X)^-1`` with the Bartlett-weighted meat matrix::

        S = sum_t s_t s_t' + sum_{j=1}^{L} w_j sum_t (s_t s_{t-j}' + s_{t-j} s_t'),
        s_t = x_t e_t,    w_j = 1 - j / (L + 1)

    Series are aligned on their shared index and rows missing any variable are dropped.
    """
    if not regressors:
        raise ValueError("need at least one regressor")

    frame = pd.concat({"__y__": y, **regressors}, axis=1).dropna()
    n = len(frame)
    names = list(regressors)
    if n < len(names) + 2:
        raise ValueError(f"need at least three paired observations, got {n}")

    lags = bartlett_lag_default(n) if lags is None else lags
    if lags < 0 or lags >= n:
        raise ValueError(f"lags must be in [0, {n - 1}], got {lags}")

    target = frame["__y__"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), *(frame[name].to_numpy(dtype=float) for name in names)])

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ target
    residuals = target - design @ coefficients

    # HAC meat matrix: the Bartlett-weighted sum of lagged score cross-products.
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)

    covariance = xtx_inv @ meat @ xtx_inv
    labels = ["const", *names]
    centred = target - target.mean()

    return HacRegression(
        coefficients=pd.Series(coefficients, index=labels, name="estimate"),
        std_errors=pd.Series(np.sqrt(np.diag(covariance)), index=labels, name="std_error"),
        r_squared=1.0 - float(residuals @ residuals) / float(centred @ centred),
        n_obs=n,
        lags=lags,
    )


def subsample_means(series: pd.Series, by: pd.Series) -> pd.DataFrame:
    """Newey-West means within each group of ``by``.

    Used to ask whether the decline in the premium across decades survives its own error bars. A
    table of point estimates invites the reader to see a trend whether or not one is there, so each
    subsample carries its own robust standard error.
    """
    frame = pd.concat({"value": series, "group": by}, axis=1).dropna()
    rows = []
    for group, chunk in frame.groupby("group", sort=True):
        estimate = newey_west_mean(chunk["value"])
        rows.append(
            {
                "group": group,
                "n": estimate.n_obs,
                "mean": estimate.mean,
                "std_error": estimate.std_error,
                "t_statistic": estimate.t_statistic,
                "ci_low": estimate.mean - 1.96 * estimate.std_error,
                "ci_high": estimate.mean + 1.96 * estimate.std_error,
            }
        )
    return pd.DataFrame(rows).set_index("group")

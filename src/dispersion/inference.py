from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def bartlett_lag_default(n_obs: int) -> int:
    if n_obs < 2:
        raise ValueError(f"need at least two observations, got {n_obs}")
    return max(1, int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


@dataclass(frozen=True)
class MeanEstimate:
    mean: float
    std_error: float
    t_statistic: float
    n_obs: int
    lags: int
    naive_std_error: float

    @property
    def inflation(self) -> float:
        return self.std_error / self.naive_std_error


def _bartlett_long_run_variance(x: np.ndarray, lags: int) -> float:
    n = x.size
    deviations = x - x.mean()

    long_run = float(deviations @ deviations) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        autocovariance = float(deviations[lag:] @ deviations[:-lag]) / n
        long_run += 2.0 * weight * autocovariance
    return long_run


def newey_west_mean(series: pd.Series, lags: int | None = None) -> MeanEstimate:
    values = series.dropna().to_numpy(dtype=float)
    n = values.size
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}")

    lags = bartlett_lag_default(n) if lags is None else lags
    if lags < 0 or lags >= n:
        raise ValueError(f"lags must be in [0, {n - 1}], got {lags}")

    long_run = _bartlett_long_run_variance(values, lags)
    if long_run <= 0.0:
        raise ValueError(f"long-run variance is {long_run:.6g}, not positive")

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


def block_bootstrap_mean(
    series: pd.Series,
    *,
    block: int = 6,
    draws: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    values = series.dropna().to_numpy(dtype=float)
    n = values.size
    if not 1 <= block < n:
        raise ValueError(f"block must be in [1, {n - 1}], got {block}")

    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(draws, n_blocks))
    index = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]
    means = values[index].mean(axis=1)

    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class ForecastRegression:
    alpha: float
    beta: float
    alpha_std_error: float
    beta_std_error: float
    r_squared: float
    n_obs: int
    lags: int

    @property
    def t_alpha(self) -> float:
        return self.alpha / self.alpha_std_error

    @property
    def t_beta_equals_one(self) -> float:
        return (self.beta - 1.0) / self.beta_std_error


def forecast_regression(
    realized: pd.Series,
    implied: pd.Series,
    lags: int | None = None,
) -> ForecastRegression:
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
    coefficients: pd.Series
    std_errors: pd.Series
    r_squared: float
    n_obs: int
    lags: int

    @property
    def t_statistics(self) -> pd.Series:
        return (self.coefficients / self.std_errors).rename("t")

    def summary(self) -> pd.DataFrame:
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

    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)

    covariance = xtx_inv @ meat @ xtx_inv
    labels = ["const", *names]
    centered = target - target.mean()

    return HacRegression(
        coefficients=pd.Series(coefficients, index=labels, name="estimate"),
        std_errors=pd.Series(np.sqrt(np.diag(covariance)), index=labels, name="std_error"),
        r_squared=1.0 - float(residuals @ residuals) / float(centered @ centered),
        n_obs=n,
        lags=lags,
    )


def subsample_means(series: pd.Series, by: pd.Series) -> pd.DataFrame:
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

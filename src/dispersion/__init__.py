"""Dispersion: implied versus realised correlation on index and constituent options."""

from dispersion.correlation import (
    average_correlation,
    index_variance,
    index_vol_from_average_correlation,
)
from dispersion.history import (
    correlation_window,
    implied_correlation_on,
    realized_correlation_history,
)
from dispersion.inference import (
    forecast_regression,
    hac_regression,
    newey_west_mean,
    subsample_means,
)
from dispersion.realized import annualized_vol, basket_return
from dispersion.strategy import (
    expanding_quantile_signal,
    gross_pnl,
    net_pnl,
    performance,
    trading_cost,
)
from dispersion.tail import (
    crash_exposure,
    event_table,
    stress_buckets,
    volatility_exposure,
)

__all__ = [
    "annualized_vol",
    "average_correlation",
    "basket_return",
    "correlation_window",
    "crash_exposure",
    "event_table",
    "expanding_quantile_signal",
    "forecast_regression",
    "gross_pnl",
    "hac_regression",
    "implied_correlation_on",
    "index_variance",
    "index_vol_from_average_correlation",
    "net_pnl",
    "newey_west_mean",
    "performance",
    "realized_correlation_history",
    "stress_buckets",
    "subsample_means",
    "trading_cost",
    "volatility_exposure",
]

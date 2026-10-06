"""Implied and realized correlation of the S&P 500 and the dispersion trade."""

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
from dispersion.realized import annualized_vol, basket_return

__all__ = [
    "annualized_vol",
    "average_correlation",
    "basket_return",
    "correlation_window",
    "implied_correlation_on",
    "index_variance",
    "index_vol_from_average_correlation",
    "realized_correlation_history",
]

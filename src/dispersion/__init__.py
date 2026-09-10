"""Implied and realized correlation of the S&P 500 and the dispersion trade."""

from dispersion.correlation import (
    average_correlation,
    index_variance,
    index_vol_from_average_correlation,
)

__all__ = [
    "average_correlation",
    "index_variance",
    "index_vol_from_average_correlation",
]

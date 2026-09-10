"""Tests for the correlation identity.

The point of these tests is not coverage, it is trust. Rung 2 onwards rests on equation (3) being
right, so the properties asserted here are the mathematical facts that make the formula meaningful,
not incidental behaviour of the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dispersion.correlation import (
    average_correlation,
    index_variance,
    index_vol_from_average_correlation,
)

# A two-asset basket small enough to check by hand.
#
#   A   = 0.5(0.20) + 0.5(0.30)         = 0.25
#   B   = 0.25(0.04) + 0.25(0.09)       = 0.0325
#   A^2 - B                             = 0.0625 - 0.0325 = 0.03
#   sigma_I^2 at rho = 0.5              = 0.0325 + 0.5(0.03) = 0.0475
#
HAND_WEIGHTS = np.array([0.5, 0.5])
HAND_VOLS = np.array([0.20, 0.30])
HAND_RHO = 0.5
HAND_INDEX_VARIANCE = 0.0475


def test_hand_computed_two_asset_case():
    """The worked example above, in both directions."""
    index_vol = index_vol_from_average_correlation(HAND_VOLS, HAND_WEIGHTS, HAND_RHO)
    assert index_vol**2 == pytest.approx(HAND_INDEX_VARIANCE)

    recovered = average_correlation(index_vol, HAND_VOLS, HAND_WEIGHTS)
    assert recovered == pytest.approx(HAND_RHO)


def test_two_assets_is_exact():
    """With one pair there is nothing to average, so equation (3) is the exact identity (1)."""
    rho_matrix = np.array([[1.0, HAND_RHO], [HAND_RHO, 1.0]])
    exact = index_variance(HAND_VOLS, HAND_WEIGHTS, rho_matrix)
    assert exact == pytest.approx(HAND_INDEX_VARIANCE)


def test_perfect_correlation_gives_weighted_average_vol():
    """rho_bar = 1 means no diversification: sigma_I = A = sum w_i sigma_i."""
    weights = np.array([0.2, 0.3, 0.5])
    vols = np.array([0.15, 0.25, 0.40])

    index_vol = index_vol_from_average_correlation(vols, weights, 1.0)
    assert index_vol == pytest.approx(float(weights @ vols))


def test_zero_correlation_gives_diagonal_only():
    """rho_bar = 0 means the cross terms vanish: sigma_I = sqrt(sum w_i^2 sigma_i^2)."""
    weights = np.array([0.2, 0.3, 0.5])
    vols = np.array([0.15, 0.25, 0.40])

    index_vol = index_vol_from_average_correlation(vols, weights, 0.0)
    assert index_vol == pytest.approx(float(np.sqrt(np.sum(weights**2 * vols**2))))


def test_diversification_is_monotone_in_correlation():
    """Index volatility rises with correlation. This is the whole economic content of the trade."""
    weights = np.array([0.4, 0.35, 0.25])
    vols = np.array([0.20, 0.30, 0.45])

    index_vols = [
        index_vol_from_average_correlation(vols, weights, rho) for rho in np.linspace(0.0, 1.0, 11)
    ]
    assert np.all(np.diff(index_vols) > 0)


def test_homogeneous_correlations_make_the_approximation_exact():
    """When every pairwise correlation is equal, equation (3) is not an approximation at all.

    All of the error in the single-correlation model comes from *dispersion in the rho_ij*. If the
    pairwise correlations are homogeneous there is nothing to lose, and the approximation must
    reproduce the exact double sum to floating-point precision.
    """
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    vols = np.array([0.18, 0.22, 0.31, 0.45])
    rho = 0.4

    n = weights.size
    rho_matrix = np.full((n, n), rho)
    np.fill_diagonal(rho_matrix, 1.0)

    exact = index_variance(vols, weights, rho_matrix)
    approximate = index_vol_from_average_correlation(vols, weights, rho) ** 2
    assert exact == pytest.approx(approximate, rel=1e-15)


def test_average_correlation_lies_between_the_pairwise_extremes():
    """rho_bar really is an average: it is a weighted mean of the pairwise correlations.

    From equation (2), rho_bar = sum_{i!=j} w_i w_j rho_ij sigma_i sigma_j / sum_{i!=j} w_i w_j
    sigma_i sigma_j. Every weight in that mean is positive, so the result cannot escape the range
    of the rho_ij it averages. This is what justifies the name.
    """
    weights = np.array([0.15, 0.25, 0.35, 0.25])
    vols = np.array([0.20, 0.28, 0.35, 0.50])

    pairwise = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])
    n = weights.size
    rho_matrix = np.eye(n)
    rho_matrix[np.triu_indices(n, k=1)] = pairwise
    rho_matrix = rho_matrix + rho_matrix.T - np.eye(n)

    index_vol = np.sqrt(index_variance(vols, weights, rho_matrix))
    rho_bar = average_correlation(index_vol, vols, weights)

    assert pairwise.min() < rho_bar < pairwise.max()


@st.composite
def baskets(draw):
    """Random valid baskets: positive weights summing to exactly 1, positive volatilities."""
    n = draw(st.integers(min_value=2, max_value=10))
    finite = {"allow_nan": False, "allow_infinity": False}

    raw = np.array(
        draw(st.lists(st.floats(0.01, 1.0, **finite), min_size=n, max_size=n)), dtype=float
    )
    weights = raw / raw.sum()
    # Force the sum to be exactly 1 in floating point, not merely close.
    weights[-1] = 1.0 - weights[:-1].sum()

    vols = np.array(
        draw(st.lists(st.floats(0.05, 1.50, **finite), min_size=n, max_size=n)), dtype=float
    )
    return weights, vols


@given(basket=baskets(), rho_bar=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=300)
def test_round_trip_for_arbitrary_baskets(basket, rho_bar):
    """average_correlation inverts index_vol_from_average_correlation, for any valid basket."""
    weights, vols = basket
    index_vol = index_vol_from_average_correlation(vols, weights, rho_bar)
    recovered = average_correlation(index_vol, vols, weights)
    assert recovered == pytest.approx(rho_bar, abs=1e-9)


@given(basket=baskets(), rho_bar=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=300)
def test_index_vol_never_exceeds_weighted_average_vol(basket, rho_bar):
    """Diversification cannot hurt: sigma_I <= A, with equality only at rho_bar = 1."""
    weights, vols = basket
    index_vol = index_vol_from_average_correlation(vols, weights, rho_bar)
    assert index_vol <= float(weights @ vols) + 1e-12


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        average_correlation(0.2, [0.2, 0.3], [0.5, 0.4])


def test_weights_and_vols_must_be_the_same_length():
    with pytest.raises(ValueError, match="length"):
        average_correlation(0.2, [0.2, 0.3, 0.4], [0.5, 0.5])


def test_single_constituent_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        average_correlation(0.2, [0.2], [1.0])


def test_negative_volatility_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        average_correlation(0.2, [0.2, -0.3], [0.5, 0.5])


def test_correlation_matrix_shape_is_checked():
    with pytest.raises(ValueError, match="must be 2x2"):
        index_variance([0.2, 0.3], [0.5, 0.5], np.eye(3))


def test_correlation_matrix_diagonal_is_checked():
    bad = np.array([[0.9, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="unit diagonal"):
        index_variance([0.2, 0.3], [0.5, 0.5], bad)


def test_impossible_negative_correlation_is_rejected():
    """A very negative rho_bar implies a negative variance, which no covariance matrix can produce."""
    with pytest.raises(ValueError, match="negative index variance"):
        index_vol_from_average_correlation([0.2, 0.3], [0.5, 0.5], -5.0)

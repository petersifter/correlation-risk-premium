# Correlation risk premium

[![CI](https://github.com/petersifter/correlation-risk-premium/actions/workflows/ci.yml/badge.svg)](https://github.com/petersifter/correlation-risk-premium/actions/workflows/ci.yml)

Implied and realized correlation of the S&P 500, and whether the dispersion trade that sells the
difference makes money.

## Average correlation

Let $w_i$ be basket weights summing to 1, $\sigma_i$ constituent volatilities, $\rho_{ij}$
pairwise correlations and $\sigma_I$ the index volatility. Since $R_I = \sum_i w_i R_i$,

```math
\sigma_I^2 = \sum_i \sum_j w_i w_j \rho_{ij} \sigma_i \sigma_j
           = \sum_i w_i^2 \sigma_i^2 + \sum_{i \ne j} w_i w_j \rho_{ij} \sigma_i \sigma_j
\qquad (1)
```

A 50-name basket has 1,225 pairwise correlations and none of them is quoted. Replacing every
$\rho_{ij}$ with one number $\bar\rho$ and solving gives

```math
\bar\rho = \frac{\sigma_I^2 - B}{A^2 - B},
\qquad A = \sum_i w_i \sigma_i,
\qquad B = \sum_i w_i^2 \sigma_i^2
\qquad (2)
```

using $\sum_{i \ne j} w_i w_j \sigma_i \sigma_j = A^2 - B$. With implied volatilities this is
implied correlation; with realized volatilities it is realized correlation.

Substituting (1) into (2) shows $\bar\rho$ is a weighted mean of the $\rho_{ij}$ with positive
weights $w_i w_j \sigma_i \sigma_j$, so it stays inside their range, and it is exact when all the
pairwise correlations are equal. Both are tested in `tests/test_correlation.py`.

This is the construction behind Cboe's implied correlation indices. The COR3M white paper defines
the index as SPX implied variance minus the implied variance of an uncorrelated portfolio of the
top 50 names, divided by the sum of pairwise weighted volatility products. That is (2) with the
diagonal term $B$ included. A common shortcut, $\sigma_I^2 / A^2$, drops it.

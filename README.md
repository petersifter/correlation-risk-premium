# Dispersion

Index implied volatility trades persistently above the weighted average of its constituents'
implied volatilities. That gap is the market's price for **correlation**, and it is tradeable:
sell index volatility, buy vega-weighted single-name volatility, and profit when the constituents
realise less correlation than was priced.

This repository measures that price, asks whether it is systematically too high, builds the
position that harvests it, and then examines what happens in the periods where the trade
famously breaks.

> **Status:** under construction. Rung 1 complete; rung 2's arithmetic complete, CRSP data layer
> outstanding.

## The build order

Each rung is completed, tested and committed before the next begins.

| Rung | | Status |
| --- | --- | --- |
| 1 | The variance identity, solved for average correlation | ✅ |
| 2 | Realised correlation from CRSP returns | arithmetic ✅, data layer next |
| 3 | Implied correlation from the OptionMetrics surface | |
| 4 | The premium: implied versus subsequently realised | |
| 5 | The position: vega-weighted straddles, delta hedging, costs | |
| 6 | The tail: February 2018, March 2020 | |

## Rung 1 — the identity

An index is a weighted basket, so its variance is a weighted double sum over the covariance
matrix of its constituents:

```
σ²_I  =  Σᵢ wᵢ²σᵢ²  +  Σᵢ≠ⱼ wᵢwⱼ ρᵢⱼ σᵢσⱼ
```

That is exact. The single modelling step in this project is to replace every pairwise `ρᵢⱼ` with
one average correlation `ρ̄` and solve for it. Writing `A = Σ wᵢσᵢ` and `B = Σ wᵢ²σᵢ²`:

```
ρ̄  =  (σ²_I − B) / (A² − B)
```

Feed it implied volatilities and you get implied correlation — what the options market charges for
correlation. Feed it realised volatilities and you get what the stocks actually did. Dispersion is
short the first and long the second.

The approximation is **exact** whenever the pairwise correlations are homogeneous; all of its error
comes from dispersion in the `ρᵢⱼ` themselves. `ρ̄` is a genuine weighted average of the pairwise
correlations, so it can never fall outside their range. Both facts are asserted in the tests.

## Rung 2 — realised correlation

The same identity, applied to a rolling window of daily returns, gives a history of how correlated
the index members actually were.

Two measurement choices are worth stating, because both have consequences.

**Volatilities are not demeaned.** Realised volatility is computed as `√(mean(r²)·252)` rather than
as a sample standard deviation. Over a 21-day window the mean daily return is almost pure noise, so
subtracting it adds variance to the estimator. More importantly, implied volatility is a pure
second-moment quantity — an option price says nothing about drift — and rung 4 compares the two
directly. Demeaning one side and not the other would be a silent mismatch. This is also what
variance swaps pay on.

**The index return is the real index, not a reconstruction.** Building the index from its own
constituents makes the identity hold by construction, which proves the code is right but measures
nothing. The genuine measurement uses the actual index series and reports the basis between the two
as a diagnostic. Rung 3 compares against SPX options, so the index leg has to be the real index.

The non-demeaning choice has a convenient consequence that the test suite exploits: when the index
is reconstructed from its constituents, the identity holds **exactly** on sample moments, because
the matrix of non-demeaned second moments *is* a covariance matrix. That gives a zero-tolerance
correctness test on the whole pipeline rather than one that has to allow for sampling error.

## Reproducing

Data comes from WRDS (CRSP, OptionMetrics) and is licensed, so no record-level data is committed
here. Reproducing the empirical rungs requires your own WRDS credentials; the queries used are
documented in the code.

```bash
uv sync
uv run pytest
```

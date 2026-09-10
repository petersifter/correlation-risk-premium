# Dispersion

Index implied volatility trades persistently above the weighted average of its constituents'
implied volatilities. That gap is the market's price for **correlation**, and it is tradeable:
sell index volatility, buy vega-weighted single-name volatility, and profit when the constituents
realise less correlation than was priced.

This repository measures that price, asks whether it is systematically too high, builds the
position that harvests it, and then examines what happens in the periods where the trade
famously breaks.

> **Status:** under construction. Rung 1 of 6 complete.

## The build order

Each rung is completed, tested and committed before the next begins.

| Rung | | Status |
| --- | --- | --- |
| 1 | The variance identity, solved for average correlation | ✅ |
| 2 | Realised correlation from CRSP returns | |
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

## Reproducing

Data comes from WRDS (CRSP, OptionMetrics) and is licensed, so no record-level data is committed
here. Reproducing the empirical rungs requires your own WRDS credentials; the queries used are
documented in the code.

```bash
uv sync
uv run pytest
```

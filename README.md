# Dispersion

Index implied volatility trades persistently above the weighted average of its constituents'
implied volatilities. That gap is the market's price for **correlation**, and it is tradeable:
sell index volatility, buy vega-weighted single-name volatility, and profit when the constituents
realise less correlation than was priced.

This repository measures that price, asks whether it is systematically too high, builds the
position that harvests it, and then examines what happens in the periods where the trade
famously breaks.

> **Status:** under construction. Rungs 1 and 2 complete and verified against CRSP.

## The build order

Each rung is completed, tested and committed before the next begins.

| Rung | | Status |
| --- | --- | --- |
| 1 | The variance identity, solved for average correlation | ✅ |
| 2 | Realised correlation from CRSP returns | ✅ |
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

## What rung 2 found

![Realised average correlation](reports/realized_correlation.png)

419 monthly windows, January 1990 to November 2024. Baskets of the 50 largest S&P 500 constituents
formed on point-in-time membership and market-cap weights, held 21 trading days forward.

| decade | windows | mean ρ | max ρ | mean single-name vol | mean index vol | mean basis |
| --- | --- | --- | --- | --- | --- | --- |
| 1990s | 120 | 0.234 | 0.596 | 26.1% | 13.1% | −0.011 |
| 2000s | 120 | 0.369 | 0.846 | 30.5% | 18.9% | −0.000 |
| 2010s | 120 | 0.385 | 0.930 | 20.5% | 13.2% | +0.001 |
| 2020s | 59 | 0.301 | 0.849 | 29.9% | 17.8% | −0.018 |

Three things worth stating.

**Correlation is not volatility.** The highest realised correlation in thirty-five years was July
2011 at 0.93 — the US debt-ceiling standoff and the European sovereign crisis — not March 2020,
which reached 0.85 at roughly twice the volatility. A dispersion book can be destroyed by a
correlation event that barely registers as a volatility event.

**The diversification benefit closes exactly when it is needed.** In the window formed 2019-11-29,
average single-name volatility was 14.0% against 7.8% for the index. In the window formed
2020-02-28, single-name volatility rose 7.2× to 101.2% while index volatility rose 12.0× to 93.5%.
The leg a dispersion seller is short moved further than the leg they are long, because correlation
tripled on top of the volatility move. That asymmetry is the trade's entire risk.

**Average correlation rose across the sample**, from 0.234 in the 1990s to 0.385 in the 2010s. Any
claim at rung 4 about a correlation premium has to contend with the fact that the quantity being
predicted is not stationary over this period.

## How rung 2 was verified

**The lookahead bias was measured, not assumed.** Re-forming each basket at the *end* of its window
rather than the start — which is what using today's membership to measure last month's volatility
amounts to — raises mean correlation by +0.0038 across a 53-month block spanning 2008–2012, in
53 of 53 windows, and drops the incomplete-name count from 3 to 1. Small, systematically positive,
present in every window, and it conceals the names that vanished. This repository forms at the
start.

**A single window is noisier than it looks.** The 21-day correlation estimate for the window formed
2010-04-30 is 0.864 with a bootstrap standard error of 0.065 and a 95% interval of [0.707, 0.964]
(2,000 resamples, 50 names). No claim at rung 4 can rest on one window, and a regression of realised
on implied correlation will carry errors-in-variables attenuation.

**The formula matches Cboe's published methodology**, verified against the COR3M white paper
v1.0.5: the index is the difference between SPX implied variance and "the implied variance of an
uncorrelated portfolio of the top 50 SPX components by market capitalization", divided by "the sum
of pairwise weighted implied volatility products". Diagonal term included — the detail most
informal write-ups drop. One step could not be read from the paper (an image) and is resolved
numerically at rung 3 against published COR1M values.

**Coverage is complete.** 419 windows for 419 available month-ends; zero skipped. Only 17 windows
lost any constituent at all, never more than two.

## Reproducing

Data comes from WRDS (CRSP, OptionMetrics) and is licensed, so no record-level data is committed
here. Reproducing the empirical rungs requires your own WRDS credentials; the queries used are
documented in the code.

```bash
uv sync
uv run pytest
```

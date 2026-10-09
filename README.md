# The correlation risk premium, and whether it can be traded

Options on the S&P 500 are expensive relative to options on the stocks inside it. The gap is the
market's price for correlation, and the trade that sells it is called dispersion: short the index
straddle, long a straddle on each constituent, sized so the position has no net volatility
exposure.

This repository measures that price monthly from 1990 to 2024, tests whether it exceeded what
correlation turned out to be, builds the trade, and charges the trade the bid-ask that was actually
quoted on the contracts it would have traded.

The premium existed. Implied correlation averaged 0.418 against subsequently realised correlation
of 0.346 over 346 monthly windows, a gap of 0.072 with a Newey-West t-statistic of 7.74, positive
in 72.3% of months.

It has largely disappeared. The gap was 0.169 in the 1990s and 0.027 since 2020, where the 95%
interval runs from −0.001 to 0.055 and therefore contains zero.

The trade does not clear its own transaction costs. Gross profit averages 1.385 volatility points
per unit of index vega against a measured half-spread of 1.354, leaving 0.030 with a t-statistic of
0.14 and a Sharpe ratio of 0.04 against a standard error of 0.19. It last covered costs in the
early 2000s.

Driessen, Maenhout and Vilkov (2009) established that correlation risk is priced in index options,
using S&P 100 options over 1996 to 2003. The contribution here is narrower and more practical: it
extends the measurement to 2024, and then asks the question their result does not answer, which is
whether the premium survives the cost of the 51-leg position needed to collect it.

## Notation

| symbol | meaning |
| --- | --- |
| `wᵢ` | weight of constituent *i* in the basket, with `Σ wᵢ = 1` |
| `σᵢ` | volatility of constituent *i*, annualised |
| `σ_I` | volatility of the index |
| `ρᵢⱼ` | correlation between constituents *i* and *j* |
| `ρ̄` | average correlation, defined below |
| `A` | weighted-average constituent volatility, `Σ wᵢσᵢ` |
| `B` | diagonal term, `Σ wᵢ²σᵢ²` |
| `ν` | vega, the change in option price per 1.00 of volatility |

Volatility points are percentage points of annualised volatility: a move from 0.20 to 0.21 is one
volatility point.

## Deriving the correlation measure

The index return is the weighted sum of its constituents, `R_I = Σ wᵢRᵢ`, so its variance is a
double sum over the covariance matrix:

```
σ²_I  =  Σᵢ Σⱼ wᵢwⱼ ρᵢⱼ σᵢσⱼ
```

Separating the diagonal, where `ρᵢᵢ = 1`:

```
σ²_I  =  Σᵢ wᵢ²σᵢ²  +  Σᵢ≠ⱼ wᵢwⱼ ρᵢⱼ σᵢσⱼ                                    (1)
```

Equation (1) is an identity. No assumption has entered.

The obstacle is the second term. An index of 50 names has 1,225 distinct pairwise correlations and
the S&P 500 has about 125,000, none of which the market quotes. The single modelling step in this
project replaces every off-diagonal `ρᵢⱼ` with one number and solves for it:

```
            σ²_I − Σᵢ wᵢ²σᵢ²
ρ̄  =  ───────────────────────────
         Σᵢ≠ⱼ wᵢwⱼ σᵢσⱼ
```

The denominator has a closed form. Since `(Σ wᵢσᵢ)² = Σᵢ wᵢ²σᵢ² + Σᵢ≠ⱼ wᵢwⱼσᵢσⱼ`, it equals
`A² − B`, giving

```
ρ̄  =  (σ²_I − B) / (A² − B)                                                 (2)
```

Equation (2) is the whole measurement. Supplied with implied volatilities it returns the
correlation the options market is charging for; supplied with realised volatilities it returns the
correlation the stocks delivered.

Two properties make the approximation defensible rather than merely convenient.

`ρ̄` is a genuine weighted average. Substituting (1) into (2) gives

```
ρ̄  =  Σᵢ≠ⱼ wᵢwⱼ ρᵢⱼ σᵢσⱼ  /  Σᵢ≠ⱼ wᵢwⱼ σᵢσⱼ
```

which is a weighted mean of the pairwise correlations with strictly positive weights. It therefore
cannot fall outside the range of the correlations it summarises, and the name is literally
accurate.

The approximation is also exact rather than approximate whenever the pairwise correlations are
homogeneous. All of its error comes from how much the `ρᵢⱼ` differ from one another, which locates
the error rather than leaving it unbounded. Both properties are asserted in `tests/test_correlation.py`.

Equation (2) is also the construction Cboe publishes its implied correlation indices from. The
COR3M white paper defines the index as the difference between SPX implied variance and "the implied
variance of an uncorrelated portfolio of the top 50 SPX components by market capitalization",
divided by "the sum of pairwise weighted implied volatility products". Those phrases are the
numerator and denominator of (2), including the diagonal term `B`, which informal write-ups
routinely drop in favour of `σ²_I / A²`.

## Data

Everything comes from WRDS. The tables, columns and filters are reproduced here because the result
cannot be checked without them.

| source | table | fields used |
| --- | --- | --- |
| CRSP daily stock file | `crsp.dsf` | `date`, `permno`, `ret`, `retx`, `prc`, `shrout` |
| CRSP daily index file | `crsp.dsi` | `date`, `sprtrn`, `vwretd` |
| S&P 500 membership | `crsp.dsp500list` | `permno`, `start`, `ending` |
| OptionMetrics surface | `optionm.vsurfd{year}` | `secid`, `date`, `days`, `delta`, `cp_flag`, `impl_volatility` |
| OptionMetrics quotes | `optionm.opprcd{year}` | `secid`, `date`, `exdate`, `delta`, `vega`, `best_bid`, `best_offer` |
| CRSP to OptionMetrics link | `wrdsapps_link_crsp_optionm.opcrsphist` | `permno`, `secid`, `sdate`, `edate`, `score` |

CRSP data comes from the Annual Update, the stable citable vintage, rather than the monthly one.
Market capitalisation is `abs(prc) × shrout`; the absolute value matters because CRSP negates the
price when it reports a bid-ask midpoint rather than a traded close.

SPX is `secid` 108105, confirmed from `optionm.secnmd` rather than assumed.

Constituents are matched to OptionMetrics at `score = 1`, the best match quality the WRDS link
table offers. Checked at seven dates spanning 1996 to 2023, that threshold covers all 50 basket
names on every one, so the strict setting costs no coverage. The link is date-ranged because both
identifiers are reused over time, and an undated join would attach the wrong company's options to a
permno after a reassignment.

### Sample construction

On the last trading day of each month the basket is fixed: the 50 largest S&P 500 members by market
capitalisation, using membership and weights as of that date. It is then held for the following 21
trading days, over which volatility and return are measured.

Fifty names by market capitalisation is Cboe's convention rather than a choice made here, and it is
also what dispersion desks trade, since the small tail of the index contributes almost no vega and
a great deal of transaction cost.

| | count |
| --- | --- |
| formation dates, 1990 to 2024 | 419 |
| windows produced | 419 |
| windows skipped | 0 |
| windows with an implied leg (1996 onward) | 346 |
| windows with measured costs | 346 |
| windows losing any constituent mid-window | 17 |
| most constituents lost in one window | 2 |

The 73 windows without an implied leg are those before OptionMetrics begins in 1996. They retain
their realised measurement, and every downstream statistic drops rows missing the columns it needs
rather than assuming the panel is rectangular.

## Measurement choices

Five choices affect the result, and each is stated with its consequence rather than left implicit.

Volatility is not demeaned. Realised volatility is `√(mean(r²) · 252)` rather than a sample
standard deviation. Over 21 observations the mean daily return is close to noise, so subtracting an
estimate of it adds variance to the estimator. The binding reason is comparability: an option price
contains no information about drift, so implied volatility is a pure second moment, and demeaning
only the realised side would compare two different quantities. This is also the convention variance
swaps settle on.

The choice has a useful consequence. The matrix of non-demeaned sample second moments is itself a
covariance matrix, so a basket reconstructed from its own constituents satisfies equation (1)
exactly on sample moments. That converts the correctness check on the entire pipeline into a
zero-tolerance assertion rather than one that must tolerate sampling error.

Both legs use price returns. SPX options are written on the price index, so the index leg uses
`sprtrn` and constituents use `retx` rather than `ret`. Measuring one side with dividends included
and the other without would bias the correlation. `ret` and `vwretd` are fetched alongside so the
sensitivity can be reported rather than asserted away.

The index volatility is the real index. Constructing the index from its own constituents makes
equation (1) hold by construction, which verifies the code and measures nothing. The published
measurement uses the actual S&P 500 series, and the difference between the two is reported per
window as a basis.

Baskets are formed before the window they are measured over. Taking today's membership and
measuring volatility over the past month conditions on which names survived the month. The
magnitude of that bias is measured rather than assumed: re-forming each basket at the end of its
window instead raises mean measured correlation by 0.0038, in 53 of 53 windows across 2008 to 2012,
and lowers the count of incomplete names from 3 to 1, concealing the disappearances that caused it.

Dropped names are counted. A constituent with no usable market capitalisation at formation, or
incomplete returns over the window, is excluded and the remaining weights renormalised. There is no
alternative when a volatility is undefined. Doing it silently is how survivorship bias enters a
result that otherwise looks clean, so both counts travel with every window.

### The implied leg

Implied volatility comes from OptionMetrics' standardised surface, which interpolates onto fixed
maturities of 10, 30, 60, 91 days and onward, and fixed deltas from ±10 to ±90 in steps of five.
Using it removes any need to fit a surface.

The 30-day, 50-delta point is Cboe's at-the-money convention and matches the measurement horizon,
since 21 trading days is close to 30 calendar days. Comparing a one-month forecast against a
two-week outcome would be a horizon mismatch of the same class as mixing price and total returns.

At exactly 50 delta the call and the put sit at slightly different strikes, so the gap between them
is local skew rather than noise. It averages 0.8 volatility points across the basket in calm
conditions and 8.0 in February 2020. The measurement uses their mean, which approximates the
forward at-the-money volatility, and both legs are retained so the choice can be tested. Across
2000, 2010 and 2020, no secid-day out of 2.5 million carried only one of the two, so the mean is
never silently one-sided.

SPX carries a surface row with a null implied volatility on 17 of 7,463 trading days, or 0.23%.
Those windows keep their realised measurement and lose only the implied leg.

### Estimators

Correlation is persistent, so the premium series is serially correlated and the usual standard
error of a mean understates its uncertainty. The long-run variance of a serially correlated series
is the sum of its autocovariances rather than its variance:

```
S  =  γ₀  +  2 Σⱼ₌₁ᴸ wⱼ γⱼ,        wⱼ = 1 − j/(L+1)
```

The `wⱼ` are Bartlett weights. They are not cosmetic: a raw truncated sum can produce a negative
variance estimate, and the taper guarantees the estimator stays non-negative (Newey and West,
1987). The standard error of the mean is `√(S/T)`, and `L` follows the automatic rule
`L = ⌊4(T/100)^(2/9)⌋` (Newey and West, 1994), which gives five lags for a few hundred monthly
observations.

Regressions use the same kernel in sandwich form, `(X'X)⁻¹ S (X'X)⁻¹` with
`S = Σ sₜsₜ' + Σⱼ wⱼ Σ (sₜsₜ₋ⱼ' + sₜ₋ⱼsₜ')` and `sₜ = xₜeₜ`. Both estimators are implemented
directly rather than called from a library, and `tests/test_inference.py` checks them against
`statsmodels` at matching lag length, so owning the implementation costs nothing in correctness.

Sharpe ratio standard errors use `se(SR) ≈ √((1 + ½SR²)/T)` with `T` in years (Lo, 2002).

## Realised correlation, 1990 to 2024

![Realised average correlation](reports/realized_correlation.png)

419 monthly windows. Mean realised correlation 0.325.

| decade | windows | mean ρ | max ρ | mean single-name vol | mean index vol |
| --- | --- | --- | --- | --- | --- |
| 1990s | 120 | 0.234 | 0.596 | 26.1% | 13.1% |
| 2000s | 120 | 0.369 | 0.846 | 30.5% | 18.9% |
| 2010s | 120 | 0.385 | 0.930 | 20.5% | 13.2% |
| 2020s | 59 | 0.301 | 0.849 | 29.9% | 17.8% |

Average correlation rose by about two thirds between the 1990s and the 2010s, which matters for
anything downstream: the quantity being forecast is not stationary over this sample.

Correlation and volatility are distinct risks, and the ranking of the extreme months shows it. The
highest reading in thirty-five years is July 2011 at 0.93, during the US debt-ceiling standoff and
the European sovereign crisis, ahead of March 2020 at 0.85 with roughly twice the volatility.

How much of a single window's reading is noise is worth knowing before any of it is interpreted. A
block bootstrap over the window formed 2010-04-30, using 2,000 resamples of the 21 daily
observations across 50 names, gives a point estimate of 0.864 with a standard error of 0.065 and a
95% interval of 0.707 to 0.964. Individual windows are far noisier than the smooth series suggests,
which is why nothing below rests on one of them.

## The premium

![The correlation risk premium](reports/correlation_premium.png)

Both legs are priced from the same basket: identical names, weights and date, with only the
volatilities differing. The top-50 proxy approximation therefore largely cancels between the two
sides instead of sitting between them as an unmeasured wedge.

Over 346 windows from January 1996 to November 2024, implied correlation averaged 0.4184 and
subsequently realised correlation 0.3464.

| | value |
| --- | --- |
| mean premium | 0.0720 |
| Newey-West standard error | 0.0093 |
| naive standard error | 0.0074 |
| inflation from serial correlation | 1.25× at 5 lags |
| t-statistic | 7.74 |
| months positive | 72.3% |

Serial correlation widens the interval by a quarter. A naive calculation would have reported 9.70.

### Is implied correlation an unbiased forecast

![Forecast bias and decay](reports/premium_inference.png)

The Mincer-Zarnowitz regression (Mincer and Zarnowitz, 1969) of the outcome on the forecast:

```
ρ̄_realised,t  =  α  +  β · ρ̄_implied,t  +  εₜ
```

| parameter | estimate | HAC standard error | test |
| --- | --- | --- | --- |
| α | 0.0324 | 0.0199 | t vs 0 = 1.63 |
| β | 0.7505 | 0.0564 | t vs 1 = −4.42 |

R² = 0.471, n = 346, 5 lags.

The intercept is not distinguishable from zero, so this is not a constant charge added to an
otherwise accurate forecast, which is how a risk premium is usually described. The slope is
significantly below one: implied correlation over-reacts, and a one-point rise in it forecasts only
three quarters of a point of realised correlation. The fitted line crosses the 45-degree line at an
implied correlation near 0.13, well below the sample mean of 0.42, so across the realised range the
premium grows with the level of implied correlation.

A measurement-error distinction is worth stating because it is easy to get backwards. The 0.065
standard error on a 21-day realised correlation sits in the dependent variable, where it inflates
residual variance and lowers R² but leaves β unbiased. Attenuation of β toward zero would require
noise in the regressor, meaning the implied side, which comes from surface interpolation and stale
quotes and is a smaller and separate problem.

### The decay

| decade | windows | premium | NW standard error | 95% interval | t |
| --- | --- | --- | --- | --- | --- |
| 1990s | 48 | 0.1692 | 0.0161 | 0.1376 to 0.2008 | 10.49 |
| 2000s | 120 | 0.0574 | 0.0110 | 0.0359 to 0.0789 | 5.22 |
| 2010s | 120 | 0.0694 | 0.0159 | 0.0382 to 0.1005 | 4.37 |
| 2020s | 58 | 0.0269 | 0.0144 | −0.0012 to 0.0551 | 1.88 |

The 1990s interval does not overlap any later decade, so the decline is not an artefact of reading
point estimates off a table. The 2020s interval contains zero. On this evidence the correlation
risk premium of the last five years cannot be distinguished from nothing, before a single
transaction cost is charged against it.

## The position and its cost

![Strategy P&L](reports/strategy_pnl.png)

Short one index straddle with vega `ν_I`, long a straddle on name *i* with vega `ν_i = ν_I wᵢ`, so
the vegas net to zero. Results are quoted per unit of index vega in volatility points, which makes
the P&L directly comparable with a bid-ask spread measured the same way and independent of book
size.

### Why the P&L is a volatility spread

A delta-hedged option earns `∫ ½ Γ S² (σ²_r − σ²_i) dt` (Bakshi and Kapadia, 2003). For an
at-the-money straddle held to expiry this collapses. Using `ν = 0.3989 S √T` and
`Γ = 0.3989 / (S σ √T)`:

```
½ Γ S² T        ≈  0.2 S √T / σ
σ²_r − σ²_i     =  (σ_r − σ_i)(σ_r + σ_i)  ≈  2σ(σ_r − σ_i)
```

Multiplying, the `σ` cancels and `0.4 S √T (σ_r − σ_i) = ν (σ_r − σ_i)`. Applying that to both legs
with `ν_i = ν_I wᵢ`, the P&L per unit of index vega is

```
Σᵢ wᵢ(σ_r,ᵢ − σ_i,ᵢ)  −  (σ_r,I − σ_i,I)  =  (A_r − σ_r,I)  −  (A_i − σ_i,I)
```

the realised volatility spread minus the implied one.

Three approximations enter here and all are first-order. The expression ignores vanna and volga,
assumes gamma is roughly constant over the holding period, and assumes continuous delta hedging. It
also uses at-the-money volatility only, so skew is excluded.

### Costs

Costs come from quoted bid-ask in `opprcd`, converted into volatility points by each contract's own
vega:

```
half-spread in volatility points  =  100 (best_offer − best_bid) / (2 ν)
```

OptionMetrics quotes vega per 1.00 of volatility rather than per percentage point, verified against
the at-the-money value `0.3989 S √T`, which reproduced a reported SPX vega of 359.96 to within 0.3%.

A straddle's half-spread in volatility points equals a single option's. The call and put double both
the dollar spread and the vega, and the two factors of two cancel, so charging the single-contract
half-spread per leg is right and doubling it would double-count.

The maturity search runs over 10 to 60 days and keeps the expiry nearest 30, rather than requiring
25 to 35 days. A tight window returns nothing before about 2012: weekly options were not yet
widespread, listed expiries were monthly, and from a month-end formation date the nearest contracts
sit roughly 15 and 50 days out. The achieved tenor is reported per window and runs 19 to 22 days in
the early sample against 29 later.

Cost is charged one way rather than round trip, because a 30-day option held 21 trading days
expires at the end of the window. Round-tripping would double it; since the base case already fails,
the conclusion does not depend on this.

### Results

| | mean, volatility points | NW standard error | t |
| --- | --- | --- | --- |
| gross | 1.385 | 0.253 | 5.48 |
| cost | 1.354 | 0.085 | 15.86 |
| net | 0.030 | 0.222 | 0.14 |

The quoted bid-ask consumes 98% of the gross.

| | value |
| --- | --- |
| Sharpe ratio | 0.04 ± 0.19 |
| hit rate | 44.2% |
| worst month | −7.54 |
| maximum drawdown | −169.3 |
| cumulative total | +10.5 |
| skew | +0.89 |
| excess kurtosis | +2.33 |
| bootstrap 95% interval for the mean | −0.263 to +0.322 |

Returns are skewed and fat-tailed, with the worst month at 2.7 standard deviations, so the
bootstrap interval is quoted alongside the t-statistic rather than relying on normality.

| period | gross | cost | net |
| --- | --- | --- | --- |
| 1995-99 | 4.933 | 2.134 | 2.799 |
| 2000-04 | 1.951 | 1.374 | 0.577 |
| 2005-09 | 0.515 | 1.536 | −1.021 |
| 2010-14 | 1.000 | 0.953 | 0.046 |
| 2015-19 | 0.255 | 1.013 | −0.758 |
| 2020-24 | 0.329 | 1.268 | −0.940 |

### Conditioning on the level of implied correlation

Since β < 1 implies the premium grows with implied correlation, trading only when implied
correlation is high is the obvious refinement. It fails, and the way it appears to succeed is worth
recording.

| threshold | windows | net | Sharpe |
| --- | --- | --- | --- |
| top quartile, full-sample threshold | 87 | 0.245 | 0.28 |
| top quartile, expanding threshold | 85 | −0.266 | −0.38 ± 0.39 |

A quantile computed over the whole sample is unavailable at the time of the trade: nobody in 1996
knew the 1996 to 2024 quartile. Recomputed with a threshold built only from history up to the
previous observation, the same rule loses money. The entire improvement was lookahead, and
conditioning done honestly makes the strategy worse.

Neither variant has a Sharpe ratio distinguishable from zero.

## What the position is exposed to

![What the strategy is exposed to](reports/tail_exposure.png)

The standing objection to any short-premium strategy is that it is short volatility with extra
steps. Testing it requires the market's direction and not only its magnitude, because a short
volatility exposure and a short crash exposure are indistinguishable in a volatility-only view.

| regressors | R² |
| --- | --- |
| index volatility risk premium | 0.060 |
| market return and its square | 0.011 |
| correlation surprise, `ρ̄_realised − ρ̄_implied` | 0.422 |

The index volatility risk premium supplies almost the entire average P&L. Of 1.385 volatility
points of gross, the short index leg contributes 1.374 and the long single-name leg 0.010:
single-name options are priced close to fair on average and index options are rich.

Yet it explains only 6% of the month-to-month variation. The two statements are consistent, and the
reconciliation is the point of the structure. Index and single-name volatility risk premia
correlate at 0.911, with a regression slope of 0.911 and an R² of 0.830. The single-name leg hedges
away nearly all of the common volatility level, and what survives is the difference between two
nearly identical premia, which is correlation.

Market exposure is absent:

| regressor | estimate | HAC standard error | t |
| --- | --- | --- | --- |
| constant | −0.038 | 0.243 | −0.16 |
| market return | 6.336 | 4.709 | 1.35 |
| market return squared | 7.602 | 46.792 | 0.16 |

A negative coefficient on the squared term would be the short-gamma signature. There is none.

The quintile table imposes no functional form, which makes it the check on the quadratic:

| quintile | windows | mean market return | mean net P&L | 95% interval | worst |
| --- | --- | --- | --- | --- | --- |
| Q1 | 70 | −6.1% | −0.078 | −1.173 to 1.016 | −7.55 |
| Q2 | 69 | −0.9% | −0.764 | −1.379 to −0.148 | −6.17 |
| Q3 | 69 | 1.3% | −0.147 | −0.956 to 0.661 | −6.69 |
| Q4 | 69 | 3.2% | 0.498 | −0.258 to 1.253 | −4.97 |
| Q5 | 69 | 6.6% | 0.644 | −0.283 to 1.571 | −6.74 |

The worst bucket by mean is Q2 rather than the crash bucket Q1, and the worst individual months are
scattered across all five.

Correlation surprise, by contrast, carries a slope of −13.08 with a t-statistic of −14.0.

The hedge therefore works, which vindicates the structure of the trade and simultaneously explains
why the net P&L is small and noisy. What remains after hedging is a thin spread between two
volatility risk premia that move almost together.

### The losses are correlation events

| episode | formed | implied ρ | realised ρ | surprise | market return | net P&L |
| --- | --- | --- | --- | --- | --- | --- |
| LTCM and Russia | 1998-07-31 | 0.455 | 0.526 | 0.07 | −14.6% | −0.42 |
| Lehman | 2008-09-30 | 0.606 | 0.762 | 0.16 | −20.3% | −3.32 |
| Euro sovereign crisis | 2011-07-29 | 0.764 | 0.930 | 0.17 | −6.4% | −2.52 |
| China devaluation | 2015-07-31 | 0.333 | 0.658 | 0.33 | −6.3% | −2.30 |
| Volmageddon | 2018-01-31 | 0.183 | 0.560 | 0.38 | −4.7% | −6.06 |
| Covid | 2020-02-28 | 0.728 | 0.849 | 0.12 | −11.1% | −2.53 |

Every episode is a loss and in every one realised correlation exceeded implied. Reading the last
two columns together, the worst month is February 2018, which had the mildest market decline of the
six at −4.7%, while Lehman's −20.3% produced roughly half that loss. Severity tracks the correlation
surprise rather than the market move. A dispersion book is not destroyed by a crash. It is destroyed
by correlation going to one, which a crash usually causes but does not require.

## Validation

The suite is 108 tests and runs on synthetic fixtures, so the mathematics can be checked without a
WRDS subscription. Four of the tests do more than guard against regressions.

The identity holds exactly on reconstructed baskets. Because volatilities are not demeaned, a
basket built from its own constituents satisfies equation (1) to floating-point precision, so the
check runs at zero tolerance rather than allowing for sampling error. A failure means the pipeline
is broken, not noisy.

No-lookahead is a property, not a claim. Truncating the input series after any date must leave
every earlier weight and signal unchanged. Both the point-in-time weights and the expanding
quantile threshold are tested this way, and a companion test pins down that a full-sample quantile
fails the same property, so the distinction cannot quietly erode.

The hand-written estimators match a reference. Newey-West means and HAC regressions are checked
against `statsmodels` at matching lag length to within 1e-10.

Expected shortages and pipeline defects raise different exception types. A window that genuinely
cannot be measured is skipped and counted; anything else propagates. Catching a bare `ValueError`
around the measurement loop would silence the guards written to detect survivorship bugs, and a
test asserts the two stay distinct.

## Limitations

These all push the same way, so the net P&L above is an upper bound.

Commissions and exchange fees are not modelled, and the position touches 51 option legs monthly.
Delta-hedging costs are not modelled either: the P&L identity assumes continuous hedging, and
hedging the underlying across 21 days costs equity spread and commission.

Before 2010 a median of 10 of the 50 names have no two-sided option quote and are dropped from the
cost calculation with the remaining weights renormalised. Those are the illiquid end of the basket,
so their spreads exceed the ones measured, meaning early costs are understated and early profits
overstated. Pulling the other way, the achieved cost tenor in that period is 19 to 22 days rather
than 30, and shorter options carry less vega, so the same dollar spread converts into a larger cost
in volatility points.

Sensitivity to the top-50 universe, the 30-day tenor, the 50-delta point and the monthly window is
untested. All four were fixed in advance to match Cboe's convention and the measurement horizon and
never varied, which rules out specification search but leaves robustness unknown.

Two strategy variants were tried, unconditional and conditional. Both are reported.

The result is specific to vega-weighted dispersion on the top 50 names. Variance-swap dispersion,
correlation swaps, and weighting schemes other than index weight are different trades with
different cost profiles.

## Running it

```bash
uv sync
uv run pytest                                   # 108 tests, no credentials needed
uv run dispersion-reproduce --username YOU      # rebuild from WRDS, then draw
uv run dispersion-reproduce --from-cache        # redraw from an existing panel
```

`dispersion-reproduce` prints every statistic quoted above, in the order it appears here, and
writes all five figures to `reports/`. No number in this document was typed by hand.

Reproducing the empirical sections needs a WRDS account with CRSP and OptionMetrics entitlements
and a `.pgpass` file, which `db.create_pgpass_file()` writes once. Licensed data is not
redistributed here and the derived panel under `data/` is gitignored.

## Layout

| file | contents |
| --- | --- |
| `correlation.py` | the variance identity and equation (2) |
| `realized.py` | volatility primitives shared by both measurements |
| `data.py` | WRDS access, point-in-time membership, weights, quoted spreads |
| `history.py` | the monthly driver that builds the panel |
| `inference.py` | Newey-West means and HAC regression |
| `strategy.py` | the position, its P&L and its costs |
| `tail.py` | exposure analysis and the episode table |
| `plots.py` | the five figures |
| `reproduce.py` | regenerates all of it |

## References

Bakshi, G. and N. Kapadia (2003). Delta-hedged gains and the negative market volatility risk
premium. *Review of Financial Studies* 16(2), 527–566.

Cboe Global Markets (2021). *Cboe Implied Correlation Index (COR3M) White Paper*, version 1.0.5.

Driessen, J., P. Maenhout and G. Vilkov (2009). The price of correlation risk: evidence from equity
options. *Journal of Finance* 64(3), 1377–1406.

Lo, A. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4), 36–52.

Mincer, J. and V. Zarnowitz (1969). The evaluation of economic forecasts. In J. Mincer (ed.),
*Economic Forecasts and Expectations*. NBER.

Newey, W. and K. West (1987). A simple, positive semi-definite, heteroskedasticity and
autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708.

Newey, W. and K. West (1994). Automatic lag selection in covariance matrix estimation. *Review of
Economic Studies* 61(4), 631–653.

# Dispersion — implied vs realized correlation

Sell index implied volatility, buy vega-weighted single-name implied volatility. The spread
between S&P 500 implied vol and the weighted average of its constituents' implied vols is the
market price of **correlation**. This repo measures that price, asks whether it is systematically
too high, and builds the trade that harvests it.

## Build order (do not skip rungs)

Each rung must be complete, tested and committed before the next one starts. No scaffolding debt:
a rung that a later rung rests on is not allowed to be provisional.

| Rung | What | Needs options data |
| --- | --- | --- |
| 0 | Options and volatility fundamentals (learning, not code) | no |
| 1 | The variance identity; solve it for average correlation | no |
| 2 | Realized correlation from CRSP returns | no |
| 3 | Implied correlation from the OptionMetrics surface | yes |
| 4 | The premium: implied vs subsequently realized correlation | yes |
| 5 | The position: vega-weighted straddles, delta hedging, costs | yes |
| 6 | The tail: Feb 2018, Mar 2020 — is this just short vol in disguise | yes |

## Environment
- `uv` manages everything. Python 3.12 pinned via `.python-version`.
- Run code with `uv run python ...`, `uv run pytest`. Never `pip install` — use `uv add`.
- pandas is **3.x**: copy-on-write is default. Chained assignment silently no-ops; assign explicitly.
- **Never call `db.raw_sql`.** It hands pandas a SQLAlchemy 1.4 `Connection`, which pandas 3 does
  not recognise, so it fails with `'Connection' object has no attribute 'cursor'`. `wrds` pins
  `sqlalchemy<2`, so upgrading is not an option. Use `dispersion.data.query` instead, which reaches
  the DBAPI connection underneath. Queries use psycopg2's `%(name)s` parameter style.

## Code style
Write code a senior quant would write by hand, not code that shows off.

- Vectorize with numpy/pandas. A Python loop over rows is a bug unless justified in a comment.
- No premature abstraction. No class where a function works.
- Type hints on public functions; skip them on obvious locals.
- Docstrings state **the math**, not the obvious. Write the formula, define the symbols, cite the
  paper if there is one.
- Name variables after the finance concept (`index_variance`, `weighted_avg_vol`), not `df2`, `tmp`.
- Notebooks are for exploration only. Anything reusable moves to `src/dispersion/` with a test.

## Non-negotiable research rules
1. **No lookahead.** A signal at time *t* uses only data available at *t*. Implied correlation on
   date *t* may only be compared to realized correlation over *(t, t+h]*.
2. **No survivorship bias.** Index constituents and weights must be point-in-time. Using today's
   S&P 500 membership for a 2015 backtest is the single easiest way to fake this result.
3. **Transaction costs modeled.** Options spreads are wide, especially single names. A dispersion
   result that ignores the bid-ask is not a result. Report turnover.
4. **Out-of-sample separation.** Walk-forward. Never tune on the test set.
5. **Multiple-testing honesty.** If N variants were tried, report N.
6. **Sanity thresholds.** Sharpe > 3 on daily data means a bug until proven otherwise.

## Reporting
Always report Sharpe, max drawdown, Calmar, turnover, hit rate, and a net-of-cost equity curve.
Always include a limitations section. Feb 2018 and Mar 2020 get their own discussion — a dispersion
book's tail behaviour is the most important thing about it.

## Data licensing (hard rule)
WRDS data — CRSP, Compustat, OptionMetrics, IBES — is licensed, not ours to redistribute.
**Never commit raw or derived record-level data to a public repo.** Publish code, aggregate
statistics, and charts only. Anyone reproducing the work supplies their own WRDS credentials.
Data file extensions are gitignored; do not override with `git add -f`.

## Commits
Commits are authored as `petersifter <petersifter@uchicago.edu>` and nothing else. Never append a
`Co-Authored-By` trailer or any other attribution line.

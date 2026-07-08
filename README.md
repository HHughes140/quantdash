# Insurance Alpha Lab

Interactive alpha-research dashboard for the US insurance complex: write a
signal as a one-line expression, backtest it in under a second across the
insurance universe (P&C, specialty/E&S, life, reinsurance, brokers, health,
title, insurtech), see every relevant number (Sharpe with Lo correction +
bootstrap CI, factor-adjusted alpha with Newey-West t-stats, IC, quintile
spreads, turnover/cost drag), decompose exposure by subsector and factor,
and log the hypothesis + verdict in a theory journal.

## Data backends

Resolved automatically by `get_source()` (`quantdash/data/source.py`):

1. **Axioma / Snowflake (read-only)** — on desks where the internal
   `snowflake_utilities` package is installed. Prices are synthetic
   total-return indexes cumulated from `AXIOMA.FUNDAMENTAL.STOCKS`
   `_1_DAY_RETURN` (WW4/SH), volume proxied by `_20_DAY_ADV`, and factor
   returns come from `AXIOMA.FUNDAMENTAL.FACTOR_RETURN` (13 WW4 style
   factors). Theories persist to the local DuckDB store since this source is
   read-only.
2. **Self-seeded Snowflake** — when SNOWFLAKE_* env vars are set.
3. **Local DuckDB** — offline fallback, seeded from yfinance + Ken French.

Force one with `QUANTDASH_SOURCE=axioma|snowflake|duckdb`.

## Quick start

```bash
cd quant-dashboard
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/seed_local.py --period 10y     # ~150 tickers + FF factors
.venv/bin/streamlit run app/Home.py
```

## Snowflake

The dashboard reads through a `DataSource` interface — local DuckDB by
default, Snowflake automatically when configured. Set:

```bash
export SNOWFLAKE_ACCOUNT=xy12345.us-east-1
export SNOWFLAKE_USER=...
export SNOWFLAKE_PASSWORD=...            # or SNOWFLAKE_PRIVATE_KEY_PATH=~/.ssh/sf_key.p8
export SNOWFLAKE_WAREHOUSE=COMPUTE_WH
export SNOWFLAKE_DATABASE=QUANT
export SNOWFLAKE_SCHEMA=PUBLIC
```

then install the connector and seed it:

```bash
.venv/bin/pip install snowflake-connector-python
.venv/bin/python scripts/seed_snowflake.py --period 10y
```

Tables created: `PRICES(date, ticker, ohlc, adj_close, volume)`,
`FACTORS(date, factor, value)` (FF5 + MOM daily, decimals),
`THEORIES(...)` (the journal). The app picks Snowflake over DuckDB whenever
`SNOWFLAKE_ACCOUNT`/`SNOWFLAKE_USER` are present (same env vars work in
`.streamlit/secrets.toml`).

## Signal DSL

Expressions evaluate over a date × ticker price panel; higher = more
attractive. Examples:

```python
rank(momentum(252, 21))                 # 12-1 momentum
-zscore(returns(5))                     # short-term reversal
rank(momentum(252, 21)) / (1 + vol(63)) # vol-adjusted momentum
where(price() > sma(200), rank(-vol(63)), rank(momentum(126, 5)))
```

Operators: `returns momentum vol sma ema price drawdown rsi volume_ratio
delay delta ts_rank ts_zscore` (time-series) · `rank zscore demean winsorize`
(cross-sectional) · `log sqrt abs sign exp clip where`.

## Layout

```
quantdash/
  data/    source.py (DuckDB + Snowflake behind one interface), seed.py, universe.py
  engine/  signals.py (DSL), backtest.py (vectorized, no lookahead),
           metrics.py (Lo Sharpe, bootstrap CI), factors.py (NW regressions)
app/       Home.py (Backtest Lab), pages/ (Theory Journal, Data Explorer)
scripts/   seed_local.py, seed_snowflake.py
```

## Methodology notes

- **No lookahead**: weights formed from the signal at close of rebalance day
  *t* earn returns from *t+1*.
- **Costs**: `cost_bps` × traded notional, charged when weights change.
- **Significance**: Lo (2002) autocorrelation-corrected Sharpe p-value and a
  block-bootstrap 95% CI; the factor tab reports Newey-West alpha t-stats.
  If factor alpha is not significant, the strategy is a factor tilt — that's
  the first thing to check before believing any backtest.

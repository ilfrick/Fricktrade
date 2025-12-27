# Strategy Models (Production Defaults)

This document describes the production-grade strategy set enabled by default and how each
model is wired into the trading loop.

## Trend-Following + Volatility Targeting
- Strategy: moving-average trend with breakout confirmation.
- Inputs: `prices`, `volumes`, `session_gain_pct`.
- Output: buy/sell/hold based on trend strength and thresholds.
- Risk overlay: position sizing scales by realized volatility vs target.

## Factor-Based Selection
- Strategy: price momentum + liquidity + low-volatility composite score.
- Inputs: recent returns, volumes, session volume.
- Output: buy when score exceeds threshold; sell/exit when it falls below.

## Stat-Arb (Pairs)
- Strategy: dynamic pair selection (rolling correlation) and z-score of spread.
- Inputs: cached price history for candidate symbols.
- Output: buy/sell when spread z-score breaches thresholds.
- Pairs refresh on a fixed interval to avoid excessive data fetch.

## Execution Models
- Algos: TWAP, VWAP, POV.
- Applied only when notional exceeds `execution.algos.min_notional`.
- TWAP/VWAP generate time-sliced child orders; POV uses estimated volume.

## Market Making
- Strategy: top-of-book quote-based quoting with inventory skew.
- Inputs: last price, spread (or configured default), inventory.
- Output: limit orders on bid/ask based on inventory target and spread.

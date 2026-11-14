<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Strategy Models (Production Defaults)

This document describes the production-grade strategy set enabled by default and how each
model is wired into the trading loop.

## Trend-Following + Volatility Targeting
- Strategy: moving-average trend with breakout confirmation, RSI filter, and volume spike confirmation.
- Inputs: `prices`, `volumes`, `session_gain_pct`.
- Filters: RSI(14) blocks overbought buys (>=70) and oversold sells (<=30). Volume confirmation requires last bar >= 1.5x average of prior 4.
- Output: buy/sell/hold based on trend strength, RSI, and volume. Includes `confidence`, `trend_strength`, `rsi`.
- Risk overlay: position sizing scales by realized volatility vs target.

## Factor-Based Selection
- Strategy: price momentum (10-bar) + liquidity + low-volatility + mean-reversion composite score.
- Inputs: recent returns, volumes, session volume.
- Mean-reversion: z-score of price vs 20-bar mean, inverted (oversold = positive).
- Trend quality gate: simplified ADX proxy — holds in choppy markets (quality < 0.3).
- Default weights: momentum 0.5, liquidity 0.2, volatility 0.1, mean-reversion 0.15.
- Output: buy when score exceeds threshold; sell/exit when it falls below. Includes `confidence`, `score`, `trend_quality`.

## Pattern Trading
- Strategy: chart pattern recognition (breakout above recent high + MA confirmation + higher highs/lows + volume spike).
- Stop loss: ATR-based adaptive stop (2x ATR with fixed `stop_loss_pct` as floor). Uses `atr()` from `app/learning/indicators.py`.
- Position management: partial take-profit, trailing stop, stop-loss exit.
- Inputs: prices, highs, lows, volumes, selection filters (price, volume, spread, catalyst).

## Stat-Arb (Pairs)
- Strategy: dynamic pair selection (rolling correlation) and z-score of spread.
- Inputs: cached price history for candidate symbols.
- Output: buy/sell when spread z-score breaches thresholds; exit when z-score reverts to near zero.
- Pairs refresh on a fixed interval to avoid excessive data fetch.

## Execution Models
- Algos: TWAP, VWAP, POV.
- Applied only when notional exceeds `execution.algos.min_notional`.
- TWAP/VWAP generate time-sliced child orders; POV uses estimated volume.

## Market Making (disabled by default)
- Strategy: top-of-book quote-based quoting with inventory skew.
- Inputs: last price, spread (or configured default), inventory.
- Output: limit orders on bid/ask based on inventory target and spread.
- Not in default config — conflicts with directional strategies.

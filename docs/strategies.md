<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Strategies

## Purpose
Generate buy/sell/hold/exit signals from market state.
Intraday signal metrics (30m/60m returns, early volume, runup/drawdown) are injected into
market state and influence actions via the signal bias guard.

## Active Strategies

- **Trend following**: `app/strategies/trend_following.py` — fast/slow MA crossover with breakout detection, **RSI(14) filter** (blocks overbought buys >=70, oversold sells <=30), and **volume confirmation** (last bar >= 1.5x average of prior 4 bars). Emits `confidence` (0-1), `trend_strength`, and `rsi`.

- **Factor model**: `app/strategies/factor_model.py` — composite score from **momentum** (10-bar lookback), **liquidity**, **volatility**, and **mean-reversion** (z-score of price vs 20-bar mean, inverted). Includes a **trend quality gate** (simplified ADX proxy): holds in choppy markets when trend quality < 0.3. Default weights: momentum 0.5, liquidity 0.2, volatility 0.1, mean-reversion 0.15. Emits `confidence`, `score`, and `trend_quality`.

- **Pattern trading**: `app/strategies/pattern_trading.py` — chart pattern recognition (breakout + MA confirmation + higher highs/lows) with **ATR-based adaptive stop** (2x ATR with fixed `stop_loss_pct` as floor). Uses `atr()` from `app/learning/indicators.py`. Includes strategy-level partial TP, trailing stop, and stop loss.

- **Stat-arb pairs**: `app/strategies/stat_arb_pairs.py` — dynamic pair selection via rolling correlation, z-score mean-reversion on spread. Pairs refresh every `refresh_minutes`. Trades when spread z-score exceeds `z_entry`, exits at `z_exit`.

## Inactive Strategies (available but disabled by default)
- RL policy (`rl_policy`), fee-aware RL policy (`rl_policy_fees`): disabled — near-uniform output (~33/33/33), needs retraining.
- Market maker (`market_maker`): conflicts with directional strategies.
- Intraday momentum (`intraday_momentum`): superseded by trend_following.

## Orchestration
- Orchestrator selects and weights strategies via `orchestrator.mode`.
- Default mode is `weight`: each strategy gets a fixed weight from `orchestrator.strategy_weights`, signals are combined by `confidence * strategy_weight`.
- RL orchestrator (`orchestrator.rl.enabled: false` by default) is available but disabled — it adds latency without conviction when untrained.
- Combine mode: `priority` or `vote` (see `strategy.combine`).
- `strategy.min_conviction`: minimum weighted score required before acting (default 0.3). Below this threshold the agent holds.

### Global Account Activity Tracking
- **Purpose**: Penalizes prolonged inactivity across all symbols on an account
- **Implementation**: `AccountActivityTracker` in `app/agents/orchestrator.py`
- **Behavior**:
  - Tracks time since last trade across all symbols per account/broker
  - No penalty within `max_idle_minutes` threshold (default: 30 minutes)
  - Applies configurable penalty for excess idle time
  - Three penalty types: `linear`, `exponential` (default), or `step`
  - Resets timer on any trade across any symbol on the account
- **Use Case**: Encourages capital efficiency in multi-symbol portfolios

## Configuration
`config/config.yaml`:
- `strategy.name` or `strategy.names` — active strategy list (default: `[trend_following, factor_model, pattern_trading, stat_arb_pairs]`)
- `strategy.combine` — signal combination mode (`priority` or `vote`)
- `strategy.min_conviction` — minimum weighted score to act (default: 0.3)
- `strategy.params.trend_following.*` — fast/slow window, breakout/exit thresholds
- `strategy.params.factor_model.*` — weights (momentum, liquidity, volatility, mr_weight), buy/sell thresholds
- `strategy.params.stat_arb_pairs.*` — lookback, z_entry, z_exit, refresh_minutes, max_pairs
- `strategy.fee_aware.*` (fee guard)
- `strategy.signal_bias_guard.*`
- `pattern_trading.*` (selection filters, pattern params, entry/risk settings)
- `orchestrator.mode` — `weight` (default), `direct`, or `select`
- `orchestrator.strategy_weights` — per-strategy weight map (default: `{trend_following: 0.35, factor_model: 0.25, pattern_trading: 0.25, stat_arb_pairs: 0.15}`)
- `orchestrator.rl.*` — RL orchestrator (disabled by default)

## Usage
- Default: set `strategy.names: [trend_following, factor_model, pattern_trading, stat_arb_pairs]` with `strategy.combine: vote`.
- Single strategy: set `strategy.name: trend_following`.
- Add confidence: strategies emit `confidence` (0-1); `_combine_signals()` uses `confidence * strategy_weight` as effective weight.

## Strategy Signal Details

### Trend Following Filters
The RSI and volume filters prevent false signals in extreme conditions:
- **RSI filter**: Computed inline over the last 14 bars. Buys are blocked when RSI >= 70 (overbought), sells when RSI <= 30 (oversold). This avoids chasing momentum that is about to reverse.
- **Volume confirmation**: A buy requires the last bar's volume to be at least 1.5x the average of the prior 4 bars. Low-volume breakouts are ignored.

### Factor Model Enhancements
- **Momentum lookback**: Uses the last 10 bars (was 3), smoothing out noise.
- **Mean-reversion factor** (`mr_weight`): Z-score of price vs its 20-bar mean, inverted (oversold = positive score). Helps catch bounce opportunities.
- **Trend quality gate**: Ratio of net price movement to total absolute movement over 14 bars (simplified ADX proxy). When < 0.3, the market is choppy and the strategy holds.

### Pattern Trading ATR Stop
The stop loss is now adaptive: `stop = max(entry - 2*ATR(14), entry * (1 - stop_loss_pct))`. The fixed percentage acts as a floor, while ATR widens the stop in volatile markets and tightens it in calm ones.

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Strategies

## Purpose
Generate buy/sell/hold/exit signals from market state.
Intraday signal metrics (30m/60m returns, early volume, runup/drawdown) are injected into
market state and influence actions via the signal bias guard.

## Active Strategies
- **Trend following**: `app/strategies/trend_following.py` — fast/slow MA crossover with breakout detection. Emits `confidence` (0-1) normalized from `trend_strength`.
- **Factor model**: `app/strategies/factor_model.py` — momentum + liquidity + volatility composite score. Emits `confidence` (0-1) normalized from `score`.
- **Pattern trading**: `app/strategies/pattern_trading.py` — chart pattern recognition with strategy-level TP/SL.

## Inactive Strategies (available but disabled by default)
- RL policy (`rl_policy`), fee-aware RL policy (`rl_policy_fees`): disabled — near-uniform output (~33/33/33), needs retraining.
- Market maker (`market_maker`): conflicts with directional strategies.
- Stat-arb pairs (`stat_arb_pairs`): optional, not in default config.
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
- `strategy.name` or `strategy.names` — active strategy list (default: `[trend_following, factor_model, pattern_trading]`)
- `strategy.combine` — signal combination mode (`priority` or `vote`)
- `strategy.min_conviction` — minimum weighted score to act (default: 0.3)
- `strategy.params.*`
- `strategy.fee_aware.*` (fee guard)
- `strategy.signal_bias_guard.*`
- `pattern_trading.*`
- `orchestrator.mode` — `weight` (default), `direct`, or `select`
- `orchestrator.strategy_weights` — per-strategy weight map (e.g., `{trend_following: 0.4, factor_model: 0.3, pattern_trading: 0.3}`)
- `orchestrator.rl.*` — RL orchestrator (disabled by default)

## Usage
- Default: set `strategy.names: [trend_following, factor_model, pattern_trading]` with `strategy.combine: vote`.
- Single strategy: set `strategy.name: trend_following`.
- Add confidence: strategies emit `confidence` (0-1); `_combine_signals()` uses `confidence * strategy_weight` as effective weight.

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Strategies

## Purpose
Generate buy/sell/hold/exit signals from market state.
Intraday signal metrics (30m/60m returns, early volume, runup/drawdown) are injected into
market state and influence actions via the signal bias guard.

## Implementations
- Intraday momentum: `app/strategies/intraday_momentum.py`.
- Trend following: `app/strategies/trend_following.py`.
- Factor model: `app/strategies/factor_model.py`.
- Stat-arb pairs: `app/strategies/stat_arb_pairs.py`.
- Market maker: `app/strategies/market_maker.py`.
- RL policy: `app/strategies/rl_policy.py`.
- Fee-aware RL policy: `app/strategies/rl_policy_fees.py`.
- Pattern trading: `app/strategies/pattern_trading.py`.

## Orchestration
- RL orchestrator consumes all strategy signals and chooses which strategy to apply.
- The RL orchestrator uses a policy-gradient update with online learning and exploration (epsilon + entropy).
- The orchestrator feature set includes `cash_pct` and `buying_power_pct` to reflect account capacity.
- Broker account flags are included in RL feature vectors to reflect broker-level blocks.
- Modes: `direct` (single strategy), `select` (top-k), or `weight` (weighted blend).
- Combine mode: `priority` or `vote` (see `strategy.combine`).

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
- `strategy.name` or `strategy.names`
- `strategy.combine`
- `strategy.params.*`
- `learning.*` (for RL)
- `strategy.fee_aware.*` (fee guard)
- `pattern_trading.*`
- `orchestrator.*`
- `orchestrator.rl.time_penalty_per_bar` - Static penalty per bar (default: 0.05)
- **Global Time Penalty:**
  - `orchestrator.rl.global_time_penalty.enabled` - Enable account-level idle tracking (default: false)
  - `orchestrator.rl.global_time_penalty.max_idle_minutes` - Idle threshold before penalties (default: 30.0)
  - `orchestrator.rl.global_time_penalty.penalty_scale` - Scale of penalty applied (default: 0.01)
  - `orchestrator.rl.global_time_penalty.penalty_type` - Type: linear, exponential, or step (default: exponential)
- `strategy.signal_bias_guard.*`

## Usage
- Single strategy: set `strategy.name: rl_policy` (or another name).
- Multiple strategies: set `strategy.names: [rl_policy, rl_policy_fees]` and choose `strategy.combine`.

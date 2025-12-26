# Strategies

## Purpose
Generate buy/sell/hold/exit signals from market state.

## Implementations
- Intraday momentum: `app/strategies/intraday_momentum.py`.
- RL policy: `app/strategies/rl_policy.py`.
- Fee-aware RL policy: `app/strategies/rl_policy_fees.py`.
- Pattern trading: `app/strategies/pattern_trading.py`.

## Orchestration
- RL orchestrator selects strategy outputs using market state, strategy signals, and AI-filter-style features.
- Combine mode: `priority` or `vote` (see `strategy.combine`).

## Configuration
`config/config.yaml`:
- `strategy.name` or `strategy.names`
- `strategy.combine`
- `strategy.params.*`
- `learning.*` (for RL)
- `strategy.fee_aware.*` (fee guard)
- `pattern_trading.*`
- `orchestrator.*`
- `orchestrator.rl.time_penalty_per_bar`

## Usage
- Single strategy: set `strategy.name: rl_policy` (or another name).
- Multiple strategies: set `strategy.names: [rl_policy, rl_policy_fees]` and choose `strategy.combine`.

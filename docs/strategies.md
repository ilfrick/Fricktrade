# Strategies

## Purpose
Generate buy/sell/hold/exit signals from market state.

## Implementations
- Intraday momentum: `app/strategies/intraday_momentum.py`.
- RL policy: `app/strategies/rl_policy.py`.
- Fee-aware RL policy: `app/strategies/rl_policy_fees.py`.
- Pattern trading: `app/strategies/pattern_trading.py`.

## Orchestration
- Simple rules or ML orchestrator select strategy outputs.
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

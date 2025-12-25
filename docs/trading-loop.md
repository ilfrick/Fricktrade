# Trading Loop

## Purpose
Runs the live trading cycle: symbol selection, signal generation, risk checks,
execution, and metrics updates.

## Implementation
- Entry: `app/agents/trader.py`.
- Loads strategies, orchestrator, and broker adapters.
- Refreshes symbols (scanner or AI filter), news cache, and open orders.
- Skips trading when markets are closed.
- Runs per-symbol signals, combines them, applies guardrails, sizes orders,
  and submits via `ExecutionEngine`.

## Key Components
- Strategy selection: `strategy.name` or `strategy.names`.
- Orchestrator: `app/agents/orchestrator.py` (RL-based selection).
- Open-order guard: `execution.open_orders`.
- Market hours gating: `app/utils/market.py`.
- Metrics: `app/monitoring/metrics.py`.

## Configuration
`config/config.yaml`:
- `strategy.*`
- `orchestrator.*`
- `execution.open_orders.*`
- `market.*`
- `news.*`
- `data.dynamic_symbols.*`
- `risk.*`
- `trading_limits.*`

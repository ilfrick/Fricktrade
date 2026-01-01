# Trading Loop

## Purpose
Runs the live trading cycle: symbol selection, signal generation, risk checks,
execution, and metrics updates.

## Implementation
- Entry: `app/agents/trader.py`.
- Loads strategies, orchestrator, broker adapters, and venue gating.
- Per-symbol venue gating uses `market.symbol_venues` and falls back to `market.default_symbol_venue`.
- Symbol venue mappings can be refreshed from the broker via `market.symbol_venues_auto`.
- Refreshes symbols (scanner or AI filter), news cache, and open orders.
- Cash-aware symbol filtering enforces price <= buying power; held/open-order symbols are retained.
- The dynamic symbol cap is limited to the tradeable universe size (plus any held/open-order symbols).
- Live market data uses `data.provider` (yfinance, alpaca, or brokers).
- Skips trading when markets are closed.
- Runs per-symbol signals, combines them, applies guardrails, sizes orders,
  and submits via `ExecutionEngine`.
- Adds intraday signal metrics (30m/60m returns, early volume, runup/drawdown) to market state
  and uses a bias guard to block counter-trend actions.
- `sell` actions are ignored when no long position exists (prevents short attempts).
- Multi-broker routing uses `execution.brokers.routing` to choose the broker per symbol and supports optional fallback when a broker is down.
- Saves periodic checkpoints of in-memory state for reboot resilience.

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
- `checkpointing.*`

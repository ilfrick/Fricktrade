# Execution

## Purpose
Place, cancel, and close orders through the active broker.

## Implementation
- `app/execution/executor.py` wraps broker methods.
- `app/agents/trader.py` manages open order guard and cancel logic.

## Configuration
`config/config.yaml`:
- `execution.open_orders.enabled`
- `execution.open_orders.skip_if_pending`
- `execution.open_orders.strategy_guard`
- `execution.open_orders.interval_seconds`

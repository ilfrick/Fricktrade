# Execution

## Purpose
Place, cancel, and close orders through the active broker.

## Implementation
- `app/execution/order_queue.py` queues orders and submits one at a time.
- `app/execution/executor.py` wraps broker methods.
- `app/agents/trader.py` manages open order guard, cancel logic, and queue updates.

## Configuration
`config/config.yaml`:
- `execution.open_orders.enabled`
- `execution.open_orders.skip_if_pending`
- `execution.open_orders.strategy_guard`
- `execution.open_orders.interval_seconds`

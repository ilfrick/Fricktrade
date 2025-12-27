# Execution

## Purpose
Place, cancel, and close orders through the active broker.

## Implementation
- `app/execution/order_queue.py` queues orders and submits one at a time.
- `app/execution/executor.py` wraps broker methods.
- `app/agents/trader.py` manages open order guard, cancel logic, and queue updates.
- `app/execution/algos.py` provides TWAP/VWAP/POV slicing helpers.

## Configuration
`config/config.yaml`:
- `execution.algos.enabled`
- `execution.algos.default`
- `execution.algos.min_notional`
- `execution.algos.twap.*`
- `execution.algos.vwap.*`
- `execution.algos.pov.*`
- `execution.open_orders.enabled`
- `execution.open_orders.skip_if_pending`
- `execution.open_orders.strategy_guard`
- `execution.open_orders.interval_seconds`

## Order Flow
1) Strategy signal -> orchestrator selection.
2) Risk checks + sizing.
3) Optional execution algo slices orders over time.
4) Order queue submits one order at a time.
5) Broker response feeds back to orchestrator.

<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Execution

## Purpose
Place, cancel, and close orders through the active broker.

## Implementation
- `app/execution/order_queue.py` queues orders and submits one at a time.
- `app/execution/executor.py` wraps broker methods.
- `app/agents/trader.py` manages open order guard, cancel logic, and queue updates.

## Configuration
`config/config.yaml`:
- `execution.symbol_executor_workers` — ThreadPool size for parallel symbol processing (default: 4).
- `execution.open_orders.enabled`
- `execution.open_orders.skip_if_pending`
- `execution.open_orders.interval_seconds`
- `execution.algos.enabled`
- `execution.algos.default`
- `execution.algos.min_notional`
- `execution.algos.adaptive.*`
- `execution.algos.twap.*`
- `execution.algos.vwap.*`
- `execution.algos.pov.*`
- `execution.impact.*`
- `execution.retry.*`

## Order Flow
1) Strategy signal -> orchestrator selection.
2) Risk checks + sizing.
3) **Pending leverage check** (buy orders only): atomic check that projected leverage (gross exposure + pending notional + new order) / equity stays under `max_portfolio_leverage`. Prevents ThreadPool race conditions where multiple workers pass the leverage check simultaneously with stale snapshots.
4) Order queue submits one order at a time.
5) In-memory portfolio `gross_exposure` updated after enqueue so subsequent threads see accurate leverage.
6) Broker response feeds back to orchestrator; pending notional released on terminal responses (completed/rejected/canceled).
7) If an execution algo is enabled and notional exceeds the threshold, orders are time-sliced.

## Position-Close Retry Bypass
Position-close orders (`is_position_close=True`) bypass the retry notional budget (`execution.retry.max_notional`). This prevents sell exits from being blocked when the daily retry budget is exhausted by failed orders. Regular buy orders still respect the budget.

## Exit Backoff
When a sell order is rejected, an exponential backoff (1/2/4/8/15 min cap) is applied to that broker+symbol pair. The agent skips exit evaluation for that pair until the backoff expires. On successful sell completion, the backoff is cleared. This prevents hammering the broker with repeated failing exit attempts.

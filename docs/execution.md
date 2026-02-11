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
3) Order queue submits one order at a time.
4) Broker response feeds back to orchestrator.
5) If an execution algo is enabled and notional exceeds the threshold, orders are time-sliced.

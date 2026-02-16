<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Execution

## Purpose
Place, cancel, and close orders through the active broker.

## Implementation
- `app/execution/order_queue.py` queues orders and submits one at a time.
- `app/execution/executor.py` wraps broker methods (supports `order_type` and `limit_price`).
- `app/execution/smart_router.py` `SmartOrderRouter` — intelligent algo selection based on order size, spread, volatility, and urgency (Almgren-Chriss inspired).
- `app/execution/tca.py` `TCAAnalyzer` — transaction cost analysis (slippage, implementation shortfall, impact).
- `app/agents/trader.py` manages open order guard, cancel logic, queue updates, limit order pricing, and TCA feedback.

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
2) Risk checks + sizing (with TCA slippage penalty).
3) **Limit order auto-upgrade**: If `order_type` is "market" and `spread_pct` is available in market_state, computes mid-price limit: `last_price ± half_spread`. Falls back to market order when no spread data.
4) **Pending leverage check** (buy orders only): atomic check that projected leverage (gross exposure + pending notional + new order) / equity stays under `max_portfolio_leverage`. Prevents ThreadPool race conditions where multiple workers pass the leverage check simultaneously with stale snapshots.
5) **SmartOrderRouter**: `_plan_execution()` delegates to `SmartOrderRouter.route()` for intelligent algo selection (TWAP/VWAP/POV/market) based on order size, spread, volatility, and urgency. Falls back to legacy algo selection on failure.
6) Order queue submits one order at a time.
7) In-memory portfolio `gross_exposure` updated after enqueue so subsequent threads see accurate leverage.
8) Broker response feeds back to orchestrator; pending notional released on terminal responses (completed/rejected/canceled).
9) **TCA feedback**: On completed fills, computes slippage_bps from fill vs decision price. Tracks per-symbol slippage penalty (EWMA α=0.3, decay 0.9x daily). Penalty reduces allowed position size by up to 50% for high-slippage symbols.

## Position-Close Retry Bypass
Position-close orders (`is_position_close=True`) bypass the retry notional budget (`execution.retry.max_notional`). This prevents sell exits from being blocked when the daily retry budget is exhausted by failed orders. Regular buy orders still respect the budget.

## Exit Backoff
When a sell order is rejected, an exponential backoff (1/2/4/8/15 min cap) is applied to that broker+symbol pair. The agent skips exit evaluation for that pair until the backoff expires. On successful sell completion, the backoff is cleared. This prevents hammering the broker with repeated failing exit attempts.

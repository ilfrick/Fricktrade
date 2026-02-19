<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Trading Loop

## Purpose
Runs the live trading cycle: symbol selection, signal generation, risk checks,
execution, and metrics updates.

## Implementation
- Entry: `app/agents/trader.py`.
- **Parallel Execution:** Symbols are processed concurrently using a `ThreadPoolExecutor` (configurable via `execution.symbol_executor_workers`, default 4).
  - Market data fetching is unlocked (parallel IO).
  - AI inference (`strategy.generate_signal`, `orchestrator.select`) is unlocked (parallel CPU).
  - Critical state updates (risk checks, order placement) are serialized via locks for safety.
- **Decoupled Reporting:** A separate daemon thread updates account metrics and market status every 15 seconds, ensuring dashboards remain responsive regardless of trading loop latency.
- Loads strategies, orchestrator, broker adapters, and venue gating.
- Per-symbol venue gating uses `market.symbol_venues` and falls back to `market.default_symbol_venue`.
- Symbol venue mappings can be refreshed from the broker via `market.symbol_venues_auto`.
- Refreshes symbols (scanner or AI filter with Keras overlay), news cache (processed via Ollama LLM gate), and open orders.
- Cash-aware symbol filtering enforces price <= buying power; held/open-order symbols are retained.
- The dynamic symbol cap is limited to the tradeable universe size (plus any held/open-order symbols).
- Live market data uses `data.provider` (yfinance, alpaca, or brokers); yfinance runs through the market-cache service (Redis + file fallback).
- When `data.dynamic_symbols.ai_filter.use_cached_symbols` is enabled, the trader consumes cached AI-filter symbol lists if available.
- Keras overlay for AI symbol filter uses TensorFlow/Keras and falls back to CPU if GPU is disabled.
- `data.process_on_new_bar_only` skips per-symbol processing when bars have not advanced.
- Skips trading when markets are closed (extended hours included when enabled).
- Runs per-symbol signals, combines them, applies guardrails, sizes orders,
  and submits via `ExecutionEngine`.
- Adds intraday signal metrics (30m/60m returns, early volume, runup/drawdown) to market state
  and uses a bias guard to block counter-trend actions.
- `sell` actions are ignored when no long position exists (prevents short attempts).
- Multi-broker routing uses `execution.brokers.routing` to choose the broker per symbol and supports optional fallback when a broker is down.
- Saves periodic checkpoints of in-memory state for reboot resilience.
- `risk.enabled: false` bypasses risk checks, but broker account flags and trading limits still block orders.
- Entry-only guards (VaR, exposure caps, order limits, cooldown, risk limits) are skipped for sell-to-close orders that reduce existing positions.
- `RiskManager.risk_preflight()` consolidates order limits + cooldown + risk limits into a single check for entry orders.
- **Two-phase symbol dispatch:** Within each broker batch, position-holding symbols
  (exits, stops, take-profits) are submitted to the ThreadPool and fully awaited
  (`concurrent.futures.wait`) before new-entry candidates are submitted. This
  guarantees that freed notional from exits is reflected in `gross_exposure` (under
  `self._lock`) before any buy evaluation begins, regardless of portfolio size or
  worker count.
- **Pending notional counter** atomically tracks in-flight buy notional per broker to prevent ThreadPool workers from racing past the leverage limit with stale portfolio snapshots.
- **Exit backoff** applies exponential backoff (1-15 min) to sell exits that fail repeatedly, preventing retry budget exhaustion.
- Position-close orders bypass the retry notional budget (`execution.retry.max_notional`) so exits are never blocked by exhausted retry capacity.

## Key Components
- Strategy selection: `strategy.name` or `strategy.names` (default: trend_following, factor_model, pattern_trading, stat_arb_pairs).
- Orchestrator: `app/agents/orchestrator.py` (weight-based strategy selection; RL mode available but disabled by default).
- Strategy signal enhancements:
  - **Trend following**: RSI(14) overbought/oversold filter + volume spike confirmation.
  - **Factor model**: 10-bar momentum, mean-reversion z-score, trend quality gate (holds in choppy markets).
  - **Pattern trading**: ATR-based adaptive stop (2x ATR with fixed % floor).
  - **Stat-arb pairs**: spread z-score mean-reversion with dynamic pair selection.
- Open-order guard: `execution.open_orders`.
- Market hours gating: `app/utils/market.py`.
- Metrics: `app/monitoring/metrics.py`.

## Modularization Roadmap
- Split the loop into explicit services: universe selection, market data, decision pipeline, risk engine, execution, and observability.
- Replace dict-heavy contracts with typed payloads (TypedDict or dataclasses) shared across services.
- Introduce a single cycle context object that carries symbols, portfolio snapshot, and feature state.
- Move per-symbol error handling into a dedicated guardrail that emits structured skip events.
- Keep the loop as orchestration glue: schedule services, apply ops-state gating, and sleep.

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

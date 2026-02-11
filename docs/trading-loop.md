<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Trading Loop

## Purpose
Runs the live trading cycle: symbol selection, signal generation, risk checks,
execution, and metrics updates.

## Implementation
- Entry: `app/agents/trader.py`.
- **Parallel Execution:** Symbols are processed concurrently using a `ThreadPoolExecutor` (default 12 workers).
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

## Key Components
- Strategy selection: `strategy.name` or `strategy.names`.
- Orchestrator: `app/agents/orchestrator.py` (weight-based strategy selection; RL mode available but disabled by default).
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

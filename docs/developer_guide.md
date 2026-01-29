<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Developer Guide

## Where to Start
- Start with `app/agents/trader.py` to understand the end-to-end loop.
- Then review `app/agents/orchestrator.py`, `app/execution/order_queue.py`, and `app/risk/manager.py`.

## Add a Strategy
1) Implement a new class in `app/strategies/` extending `app/strategies/base.py`.
2) Wire it into `TradingAgent` in `app/agents/trader.py`.
3) Add params under `strategy.params.*` in `config/config.yaml`.

## Add a Broker
1) Implement `app/brokers/base.py` interface.
2) Wire it into `_build_broker` in `app/main.py`.
3) Add config under `brokers.*` in `config/config.yaml`.

## AI Symbol Filter
- PPO-based filter lives in `app/data/ai_filter.py`.
- Config is under `data.dynamic_symbols.ai_filter.*` in `config/config.yaml`.
- Supports Keras return overlay for advanced scoring.
- Model output is a ranked symbol list; online updates run on refresh.
- Filtered symbol cache is written per symbol (Redis + file) when enabled.

## Concurrency & Thread Safety
- **Parallel Execution:** `TradingAgent` processes symbols in parallel using a `ThreadPoolExecutor` (default 12 workers) to maximize throughput during data fetching and AI inference.
- **Thread Safety:** 
  - `OrderQueue` (`app/execution/order_queue.py`) is thread-safe via internal locks.
  - `RiskManager` (`app/risk/manager.py`) protects critical state (daily loss, drawdowns) with locks.
  - `TradingAgent` protects shared state (broker state, open orders cache) with a reentrant lock (`self._lock`).
- **Decoupled Reporting:** Account metrics and market status are updated in a separate daemon thread (`_run_reporting_loop`) to ensure observability even if the trading loop is under heavy load.

## Tests
- Unit tests: `docker compose run --rm tests-when-closed python -m pytest`.
- Backtest: `docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml`.

## Docs
- System map: `docs/system_map.md`.
- Operator guide: `docs/operator_guide.md`.
- Subsystems: `docs/` index in `docs/README.md`.

## Subsystem Structure (paths)
- Agents: `app/agents/trader.py` (loop) and `app/agents/orchestrator.py` (strategy selection).
- Strategies: `app/strategies/` (rule-based + RL) wired in `strategy.params.*`.
- Risk: `app/risk/manager.py`, `app/risk/haircut.py` (caps, VaR/CVaR, haircuts, kill switches).
- Data: `app/data/` (scanner, AI symbol filter, news, Ollama for LLM gate) and `app/utils/market.py` (market hours).
- Execution: `app/execution/` (order sizing/algos/queues, multi-broker routing).
- Learning: `app/learning/` (env, features, training, drift, registry).
- Monitoring/API: `app/api/`, `app/monitoring/metrics.py`, `docs/monitoring.md`.
- GPU fallback: `app/utils/gpu_state.py` disables GPU until restart when CUDA errors occur.

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
- **Parallel Execution:** `TradingAgent` processes symbols in parallel using a `ThreadPoolExecutor` (default 12 workers) to maximize throughput during data fetching and AI inference. Each thread receives a `copy.deepcopy` of `market_state` to prevent cross-thread mutation.
- **Thread Safety:**
  - `OrderQueue` (`app/execution/order_queue.py`) is thread-safe via internal locks. Uses `heapq` for O(log n) priority insertion.
  - `RiskManager` (`app/risk/manager.py`) protects all state mutations with locks. Includes auto-reset of daily loss at market open.
  - `TradingAgent` protects shared state (broker state, open orders cache) with a reentrant lock (`self._lock`).
  - `RLStrategyOrchestrator` stores experience buffer tensors on CPU to prevent GPU memory leaks.
- **Decoupled Reporting:** Account metrics and market status are updated in a separate daemon thread (`_run_reporting_loop`) to ensure observability even if the trading loop is under heavy load.

## Tests
- Unit tests: `docker compose run --rm tests-when-closed python -m pytest`.
- Backtest: `docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml`.

## Docs
- System map: `docs/system_map.md`.
- Operator guide: `docs/operator_guide.md`.
- Subsystems: `docs/` index in `docs/README.md`.

## Subsystem Structure (paths)
- Agents:
  - `app/agents/trader.py` — main trading loop (orchestrates all modules below).
  - `app/agents/orchestrator.py` — RL strategy selection.
  - `app/agents/symbol_manager.py` — symbol selection, AI filter, venue/sector mapping, universe resolution.
  - `app/agents/market_state.py` — typed `MarketState` dataclass.
  - `app/agents/performance.py` — `PerformanceTracker` (trade recording, stats, kill switch).
  - `app/agents/open_orders.py` — `OpenOrderManager` (order cache, pending checks, metrics).
  - `app/agents/account_metrics.py` — `AccountMetricsUpdater` (equity, drawdown, VaR/CVaR).
- Strategies: `app/strategies/` (rule-based + RL) wired in `strategy.params.*`.
- Risk:
  - `app/risk/manager.py` — core limits, cooldown, exposure caps, order limits.
  - `app/risk/config.py` — `RiskConfig` dataclass (typed config with `from_dict`/`to_dict`).
  - `app/risk/haircut.py` — stress/liquidity haircuts.
- Data: `app/data/` (scanner, AI symbol filter, news, Ollama for LLM gate) and `app/utils/market.py` (market hours).
- Utilities:
  - `app/utils/structured_log.py` — `StructuredLogger` (JSON trade/risk event logs).
  - `app/utils/volatility.py` — shared realized volatility calculation.
  - `app/utils/gpu_state.py` — GPU fallback (disables GPU until restart on CUDA errors).
- Execution: `app/execution/` (order sizing/algos/queues, multi-broker routing).
- Learning: `app/learning/` (env, features, training, drift, registry).
- Monitoring/API: `app/api/`, `app/monitoring/metrics.py`, `docs/monitoring.md`.

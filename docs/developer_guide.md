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
- Model output is a ranked symbol list; online updates run on refresh.

## Tests
- Unit tests: `docker compose run --rm tests-when-closed python -m pytest`.
- Backtest: `docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml`.

## Docs
- System map: `docs/system_map.md`.
- Operator guide: `docs/operator_guide.md`.
- Subsystems: `docs/` index in `docs/README.md`.

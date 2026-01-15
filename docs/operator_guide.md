<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Operator Guide

## Start and Stop
- Full stack: `docker compose up -d --build`
- Stop stack: `docker compose down`
- Minimal services (closed markets): autoheal, healthwatch, prometheus, daily-report, tests-when-closed.

## Health Checks
- API health: `http://localhost:18081/health`
- Grafana: `http://localhost:3002`
- Prometheus: `http://localhost:9090`

## Market Gating and Ops State
- Market hours config: `market.*` in `config/config.yaml`.
- Healthwatch scheduler: `healthwatch.market_shutdown.*`.
- Ops state file (when enabled): `/data/system_state.json`.

## Trading Safety Controls
- Risk limits: `risk.*` and `trading_limits.*`.
- Order queue guardrails: `execution.open_orders.*`.
- Strategy performance kill switch: `strategy.performance.*`.
- Manual kill switches: `kill_switch.*`.

## Data and Symbols
- Live data provider: `data.provider` (yfinance/alpaca/brokers). yfinance uses batched caching.
- Dynamic symbols: `data.dynamic_symbols.*`.
- AI symbol filter: `data.dynamic_symbols.ai_filter.*` (PPO-based).

## Models and Artifacts
- Trading PPO policy: `/app/models/ppo_policy.zip`.
- AI symbol filter PPO: `/data/ai_symbol_filter.zip` with `.meta.json`.
- Orchestrator model: `/data/orchestrator_model.pt`.

## Logs and Audit
- Trader logs: `docker compose logs -f trader`.
- Audit/compliance logs: `monitoring.audit.*` and `monitoring.compliance.*`.

## Daily Reporting Email
- Daily report email uses Alertmanager SMTP settings from `.env` (do not commit secrets).
- Required keys are listed in `.env.example`.

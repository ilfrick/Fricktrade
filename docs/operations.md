<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Operations

## Daily Checks
- `http://localhost:18081/health`
- Grafana dashboards for orders, PnL, positions, and broker status.
- Trader logs for market-open gating and AI filter status.
- Market-cache logs for yfinance fetch cadence and cache refresh.
- If `risk.enabled: false`, confirm broker account flags still block orders.

## Checkpointing and Resilience
- Checkpoint settings: `checkpointing.*` in `config/config.yaml`.
- State is saved under `checkpointing.dir` (default `/data/checkpoints`).
- Old checkpoint history is pruned by age and count.

## Healthwatch and Autoheal
- Healthwatch probes core services and exposes Prometheus metrics.
- Alerts fire if services are down or flapping.
- Autoheal restarts containers that fail health checks.
- `healthwatch.market_shutdown.keep_services` controls which services stay up when markets are closed.
- `tests-when-closed` is kept running during market shutdown to execute its closed-market suite.
- Healthwatch writes the ops state file (`healthwatch.market_shutdown.state_path`) for other services to consume.
- Alertmanager SMTP and recipient settings are sourced from `.env` and must not be committed.

## Models and Artifacts
- RL models: `/app/models` and `/data` (Docker volume).
- Active model pointer: `/app/models/model_active.json` when registry is enabled.
- Registry artifacts: `/app/models/registry/*`.
- Orchestrator checkpoints: `orchestrator.rl.*_path`.
- AI filter model: `data.dynamic_symbols.ai_filter.model_path`.

## Logs
- Use `docker compose logs -f trader`.
- Use `docker compose logs -f market-cache` for yfinance cache refresh status.
- Watch AI filter heartbeat and order queue messages.
- Audit/compliance logs rotate based on `monitoring.*.retention_days`.
- Cache staleness behavior is controlled by `market_cache.max_age_multiplier` and `market_cache.ignore_staleness`.

## Safe Restart
```bash
docker compose up -d --build
```
State will reload from checkpoints when available.

## Storage Hygiene
- Checkpoints are pruned automatically.
- Data and models live in Docker volumes; back them up as needed.

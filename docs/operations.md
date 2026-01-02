<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Operations

## Daily Checks
- `http://localhost:18081/health`
- Grafana dashboards for orders, PnL, positions, and broker status.
- Trader logs for market-open gating and AI filter status.

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

## Models and Artifacts
- RL models: `/app/models` and `/data` (Docker volume).
- Orchestrator checkpoints: `orchestrator.rl.*_path`.
- AI filter model: `data.dynamic_symbols.ai_filter.model_path`.

## Logs
- Use `docker compose logs -f trader`.
- Watch AI filter heartbeat and order queue messages.
- Audit/compliance logs rotate based on `monitoring.*.retention_days`.

## Safe Restart
```bash
docker compose up -d --build
```
State will reload from checkpoints when available.

## Storage Hygiene
- Checkpoints are pruned automatically.
- Data and models live in Docker volumes; back them up as needed.

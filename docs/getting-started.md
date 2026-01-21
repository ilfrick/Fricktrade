<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Getting Started

## Goal
Get a working local deployment, confirm health, and understand the core config.

## Prerequisites
- Docker + Docker Compose
- Alpaca credentials for live/paper trading (optional for backtests)
- Optional GPU with NVIDIA container runtime

## Setup
1) Copy env template:
```bash
cp .env.example .env
```

2) Set credentials in `.env`:
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `FRICKTRADE_API_TOKEN` (if `api.auth.enabled` in `config/config.yaml`)

3) Start services:
```bash
docker compose up -d --build
```

## Verify
- API health: `http://localhost:18081/health`
- Config UI: `http://localhost:18081/ui`
- Grafana: `http://localhost:3002`
- Ollama: `http://localhost:11434` (Ollama UI, check if it's running)

## First Backtest
```bash
docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml
```

## Next Steps
- Configure strategies and risk limits in `config/config.yaml`.
- Review `docs/trading-loop.md`, `docs/strategies.md`, `docs/risk.md`.

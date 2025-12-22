## Autotrader Agent Guide

This repo contains a Python intraday trading agent for US and EU equities (NYSE, Nasdaq, Borsa Italiana), with broker adapters, risk controls, backtesting, data download, and metrics/monitoring.

## Quick Orientation

- `app/main.py` is the CLI entrypoint with subcommands: `trade`, `backtest`, `download`, `api`, `train`, `online-train`, `evaluate`, `ingest`.
- Core loop: `app/agents/trader.py` + `app/strategies/intraday_momentum.py` + `app/execution/executor.py` + `app/risk/manager.py`.
- Broker adapters: `app/brokers/alpaca.py`, `app/brokers/ibkr.py`, abstract base in `app/brokers/base.py`.
- Backtesting: `app/backtest/engine.py` uses `backtrader` and a simple SMA strategy.
- Learning (RL): `app/learning/` for env, data loading, training, and online updates; `app/strategies/rl_policy.py` for inference.
- Data download: `app/data/downloader.py` uses `yfinance` with retry and rate limiting.
- API: `app/api/server.py` (FastAPI) with `/health`, `/config`, `/config/raw`, `/config/update`, `/restart`, and `/ui`.
- Metrics: `app/monitoring/metrics.py` exposes Prometheus counters/gauges.
- Runtime config: `config/config.yaml` (supports `${ENV_VAR}` interpolation).
- Market-hours gating: `app/utils/market.py` checks NYSE, Nasdaq, and Borsa Italiana based on `market.venues`.

## Running (Docker-first)

1) Copy env template and fill credentials:

```bash
cp .env.example .env
```

2) Start services:

```bash
docker compose up -d --build
```

Services:
- `trader`: trading loop (Prometheus metrics on port `8001`).
- `api`: FastAPI config/health (mapped to host port `18081`).
- `prometheus`: metrics scrape.
- `grafana`: dashboards (mapped to host port `3002`).
- `calendar-updater`: weekly holiday refresh (configurable).

Web UI:
- `http://localhost:18081/ui` to edit YAML config, see parameter descriptions, and request a restart.

## Useful Commands

- Download data:

```bash
docker compose run --rm trader python -m app.main download --config /app/config/config.yaml --symbols AAPL MSFT
```

- Backtest:

```bash
docker compose run --rm trader python -m app.main backtest --config /app/config/config.yaml
```

- Trade:

```bash
docker compose run --rm trader python -m app.main trade --config /app/config/config.yaml
```

- Train RL policy (offline):

```bash
docker compose run --rm trader python -m app.main train --config /app/config/config.yaml
```

- Evaluate policy and regenerate charts:

```bash
docker compose run --rm trader python -m app.main evaluate --config /app/config/config.yaml
```

- Online updates (separate process):

```bash
docker compose run --rm learner
```

- GPU online updates (requires NVIDIA Docker runtime):

```bash
docker compose --profile gpu up -d learner-gpu
```

GPU check:

```bash
docker compose exec -T learner-gpu python3 - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

- Holiday calendar refresh (one-off):

```bash
docker compose run --rm calendar-updater python -m app.utils.holiday_update --config /app/config/config.yaml --once
```

- Ingest data from configured sources:

```bash
docker compose run --rm trader python -m app.main ingest --config /app/config/config.yaml
```

- Start API:

```bash
docker compose run --rm api
```

- GPU backtest:

```bash
./scripts/backtest_gpu.sh
```

## Configuration Notes

- Risk and strategy parameters live in `config/config.yaml`.
- Learning config lives under `learning` (enable policy, guardrail mode, feature set, and optional online updates). `learning.device: auto` uses CUDA if available.
- Training writes a JSON report at `learning.training.report_path` and charts in `learning.training.report_plot_dir`.
- Models and reports are stored in `./models` via the Docker volume.
- `learning.training.resume` controls whether training resumes from an existing model or starts fresh.
- Data ingestion sources are configured under `data.sources`.
- Alpaca keys come from `ALPACA_API_KEY` / `ALPACA_API_SECRET` in `.env`.
- `brokers.ibkr.enabled` controls IBKR adapter selection. If `false`, Alpaca is used.
- Data directory is `/data` inside containers (mapped to `./data` on host).
- `data.interval` and `data.lookback_days` are clamped for yfinance intraday limits.
- `market.open_mode` chooses whether any or all configured venues must be open to trade.
- `market.venues[].holidays` is refreshed by `calendar-updater` (or can be edited manually).
- `calendar-updater` refreshes holiday calendars weekly from online sources (NYSE, Nasdaq, Italy public holidays).

## Behavior Details

- Trading loop pulls prices from yfinance in `app/main.py` for live trade mode, iterating over every symbol in `data.symbols` each cycle.
- Trading is paused when all configured markets are closed.
- Strategy emits `buy`, `sell`, `exit`, or `hold`; `exit` closes the position.
- Risk checks are threshold-based and order sizing is cash-aware using broker equity/cash plus exposure caps.
- Backtest engine loads the first matching CSV in `backtest.data_dir`.
- API `/config` masks Alpaca keys before returning; `/config/update` accepts YAML updates and `/restart` triggers a graceful container restart.
- Grafana auto-provisions the "Autotrader Overview" dashboard with trade counts/rates, PnL, and drawdown.
- Dashboard also shows active symbols and account equity/cash/invested from broker account data. Skipped orders are available via `orders_skipped_total` metrics.

## Extending the Codebase

- New strategies should extend `app/strategies/base.py` and be wired in `app/agents/trader.py`.
- New brokers should implement `app/brokers/base.py` and be added to `_build_broker` in `app/main.py`.
- Additional metrics belong in `app/monitoring/metrics.py`.

## Testing

No automated tests are present. For changes, prefer manual checks via:
- `python -m app.main backtest` in Docker.
- `/health` and `/config` endpoints via the `api` service.

## Autotrader Agent Guide

This repo contains a Python intraday trading agent for EU equities (Borsa Italiana), with broker adapters, risk controls, backtesting, data download, and metrics/monitoring.

## Quick Orientation

- `app/main.py` is the CLI entrypoint with subcommands: `trade`, `backtest`, `download`, `api`.
- Core loop: `app/agents/trader.py` + `app/strategies/intraday_momentum.py` + `app/execution/executor.py` + `app/risk/manager.py`.
- Broker adapters: `app/brokers/alpaca.py`, `app/brokers/ibkr.py`, abstract base in `app/brokers/base.py`.
- Backtesting: `app/backtest/engine.py` uses `backtrader` and a simple SMA strategy.
- Learning (RL): `app/learning/` for env, data loading, training, and online updates; `app/strategies/rl_policy.py` for inference.
- Data download: `app/data/downloader.py` uses `yfinance` with retry and rate limiting.
- API: `app/api/server.py` (FastAPI) with `/health` and `/config`.
- Metrics: `app/monitoring/metrics.py` exposes Prometheus counters/gauges.
- Runtime config: `config/config.yaml` (supports `${ENV_VAR}` interpolation).

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
- `api`: FastAPI config/health (mapped to host port `8002`).
- `prometheus`: metrics scrape.
- `grafana`: dashboards (mapped to host port `3002`).

## Useful Commands

- Download data:

```bash
docker compose run --rm trader python -m app.main download --config /app/config/config.yaml --symbols ENI.MI
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
- Learning config lives under `learning` (enable policy, guardrail mode, feature set, and optional online updates).
- Training writes a JSON report at `learning.training.report_path` and charts in `learning.training.report_plot_dir`.
- Models and reports are stored in `./models` via the Docker volume.
- Data ingestion sources are configured under `data.sources`.
- Alpaca keys come from `ALPACA_API_KEY` / `ALPACA_API_SECRET` in `.env`.
- `brokers.ibkr.enabled` controls IBKR adapter selection. If `false`, Alpaca is used.
- Data directory is `/data` inside containers (mapped to `./data` on host).
- `data.interval` and `data.lookback_days` are clamped for yfinance intraday limits.

## Behavior Details

- Trading loop pulls prices from yfinance in `app/main.py` for live trade mode.
- Strategy emits `buy`, `sell`, `exit`, or `hold`; `exit` closes the position.
- Risk checks are basic thresholds only; no PnL accounting is wired into execution.
- Backtest engine loads the first matching CSV in `backtest.data_dir`.
- API `/config` masks Alpaca keys before returning.

## Extending the Codebase

- New strategies should extend `app/strategies/base.py` and be wired in `app/agents/trader.py`.
- New brokers should implement `app/brokers/base.py` and be added to `_build_broker` in `app/main.py`.
- Additional metrics belong in `app/monitoring/metrics.py`.

## Testing

No automated tests are present. For changes, prefer manual checks via:
- `python -m app.main backtest` in Docker.
- `/health` and `/config` endpoints via the `api` service.

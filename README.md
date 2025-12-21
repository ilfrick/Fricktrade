# Autotrader (Borsa Italiana)

An intraday trading agent for EU equities with shorting support, Alpaca + IBKR integration, configurable risk controls, backtesting, local data download (yfinance), and Grafana monitoring.

## Features

- Intraday strategy engine with shorting support
- Broker adapters: Alpaca (paper/live) + Interactive Brokers (paper/live)
- Risk manager with configurable limits and circuit breakers
- Backtesting on locally downloaded data
- Data download via yfinance (Borsa Italiana tickers with .MI)
- Prometheus metrics + Grafana dashboard
- Optional CUDA acceleration for analytics/backtests (profile `gpu`)

## Repository Layout

- `app/`: core trading agent code
- `config/config.yaml`: all risk/trading parameters
- `docker/`: Dockerfiles (CPU + GPU)
- `grafana/`: Grafana provisioning + dashboard
- `prometheus/`: Prometheus scrape config
- `scripts/`: helper scripts

## Quick Start (Docker)

1) Copy env template:

```bash
cp .env.example .env
```

2) Set broker credentials in `.env`:

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`

3) Start services:

```bash
docker compose up -d --build
```

4) Open Grafana:

- URL: `http://localhost:3000`
- User: `admin`
- Pass: `admin`

## Configuration (Risk + Trading)

All critical parameters live in `config/config.yaml`.

Key knobs:

- `risk.max_daily_loss_pct`
- `risk.max_position_size_pct`
- `risk.max_portfolio_leverage`
- `risk.max_short_exposure_pct`
- `risk.max_positions`
- `risk.cooldown_seconds`
- `risk.hard_stop_pct`
- `risk.trailing_stop_pct`
- `risk.circuit_breaker_drawdown_pct`
- `strategy.params.*`

You manage strategy and risk by editing `config/config.yaml` and restarting the trader container.
The `api` service exposes a read-only config endpoint at `http://localhost:8000/config` for UI tooling.

## Notes on Borsa Italiana

IBKR provides broad EU equity access including Borsa Italiana. Alpaca does not generally support EU stocks; use Alpaca for US markets or paper testing.

## Data Download (yfinance)

```bash
./scripts/download_data.sh
```

Data is saved to `/data` inside the container (mapped to `./data`).

## Backtesting

```bash
docker compose run --rm trader python -m app.main backtest --config /app/config/config.yaml
```

### GPU Backtesting (CUDA)

```bash
./scripts/backtest_gpu.sh
```

Requires NVIDIA Docker runtime. If you don’t have a GPU, ignore this.

## Live/Paper Trading

- Alpaca: set `brokers.alpaca.base_url` to paper/live
- IBKR: set `brokers.ibkr.host/port/client_id`

Run trading loop:

```bash
docker compose run --rm trader python -m app.main trade --config /app/config/config.yaml
```

## Monitoring

Prometheus scrapes `trader:8001/metrics`.
Grafana auto-provisions a dashboard with:

- Trades total
- PnL %
- Drawdown %

## Deployment

- Production: run `docker compose up -d --build` on a server
- Ensure broker API connectivity and correct timezone
- Adjust risk parameters before live trading

## Security and Safety

- Keep API keys in `.env` only
- Start in paper trading mode
- Set conservative risk limits
- Use circuit breakers and trailing stops

## Industrial-Grade Protections Included

- Daily loss limits and circuit breakers
- Max position sizing and leverage caps
- Short exposure limits
- Cooldown windows between trades
- Trailing/hard stop settings (configurable)

## GitHub Repo Creation

Use the following steps to create a private repo and push:

```bash
git init
git add .
git commit -m "Initial Autotrader"

git remote add origin https://github.com/ilfrick/Autotrader.git

git push -u origin main
```

If you use the GitHub CLI:

```bash
gh repo create Autotrader --private --source . --remote origin --push
```

## Disclaimer

This software is for research/education. You are responsible for compliance with regulations, broker requirements, and all trading risk.

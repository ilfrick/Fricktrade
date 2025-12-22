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

If Yahoo blocks the container, set a proxy and slow down requests in `config/config.yaml`:

- `data.proxy` (e.g. `http://user:pass@proxy:8080`)
- `data.rate_limit_seconds` (e.g. `5`)

## Backtesting

```bash
docker compose run --rm trader python -m app.main backtest --config /app/config/config.yaml
```

## Learning (RL)

Train a PPO policy on local OHLCV data:

```bash
docker compose run --rm trader python -m app.main train --config /app/config/config.yaml
```

Enable learning in `config/config.yaml` by setting `learning.enabled: true`. A rule-based guardrail is configurable under `learning.guardrail`.
Training produces a report at `learning.training.report_path` with return, Sharpe, and drawdown metrics, plus charts in `learning.training.report_plot_dir`.
Models and reports are persisted under `./models` on the host.

Evaluate an existing model and regenerate charts:

```bash
docker compose run --rm trader python -m app.main evaluate --config /app/config/config.yaml
```

### Online Updates

Run online updates in a separate process:

```bash
docker compose run --rm learner
```

### Data Ingestion

Pull data from configured sources (`yfinance`, `stooq`, `alphavantage`):

```bash
docker compose run --rm trader python -m app.main ingest --config /app/config/config.yaml
```

### Config Reference (Learning + Data)

```yaml
learning:
  enabled: false
  model_path: "/app/models/ppo_policy.zip"
  device: "auto"
  window_size: 50
  features:
    include_returns: true
    sma_periods: [5, 20]
    ema_periods: [10]
    rsi_periods: [14]
  guardrail:
    enabled: true
    mode: "confirm"
    params:
      lookback_minutes: 30
      entry_threshold_pct: 0.8
      exit_threshold_pct: 0.4
      allow_shorts: true
  online:
    enabled: false
    update_interval_minutes: 60
    timesteps: 1000
    eval_split: 0.1
  training:
    data_dir: "/data"
    interval: "1m"
    timesteps: 200000
    initial_cash: 100000
    commission_pct: 0.05
    slippage_bps: 2
    eval_split: 0.2
    report_path: "/app/models/training_report.json"
    report_plot_dir: "/app/models/reports"

data:
  output_dir: "/data"
  sources:
    - provider: yfinance
      enabled: true
      symbols: ["ENI.MI", "ISP.MI"]
      interval: "1m"
      lookback_days: 7
      rate_limit_seconds: 2
    - provider: stooq
      enabled: false
      symbols: ["eni", "pkn"]
      interval: "1d"
      rate_limit_seconds: 2
    - provider: alphavantage
      enabled: false
      api_key: "${ALPHAVANTAGE_API_KEY}"
      symbols: ["AAPL", "MSFT"]
      interval: "1d"
      rate_limit_seconds: 15
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

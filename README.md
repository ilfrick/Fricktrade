# Autotrader (Multi-Market)

An intraday trading agent with shorting support, Alpaca + IBKR integration, configurable risk controls, backtesting, local data download (yfinance), and Grafana monitoring. Supports multi-market trading gates (NYSE, Nasdaq, Borsa Italiana).

## Features

- Intraday strategy engine with shorting support
- Broker adapters: Alpaca (paper/live) + Interactive Brokers (paper/live)
- Risk manager with configurable limits and circuit breakers
- Cash-aware position sizing based on broker equity/cash and exposure caps
- Pattern Trading strategy for momentum breakouts with configurable filters
- Backtesting on locally downloaded data
- Data download via yfinance (US and EU tickers supported)
- Prometheus metrics + Grafana dashboard
- Optional CUDA acceleration for analytics/backtests (profile `gpu`)
- Market-open gating for NYSE, Nasdaq, and Borsa Italiana (configurable)

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

By default the `trader` service uses the GPU-enabled image. To force CPU execution for RL
inference/training, set `learning.device: cpu` in `config/config.yaml` (or via the web UI).
To disable GPU acceleration in backtests, set `backtest.use_gpu: false`.
```

4) Open Grafana:

- URL: `http://localhost:3002`
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

You manage strategy and risk by editing `config/config.yaml` and restarting the trader container, or use the web UI.
Multi-strategy mode is enabled by default via `strategy.names` (intraday momentum + pattern trading + RL policy) and can be adjusted as needed. Combine behavior is set with `strategy.combine: priority|vote`.
The `api` service exposes:
- Web UI at `http://localhost:18081/ui`
- Read-only config at `http://localhost:18081/config`
- YAML config at `http://localhost:18081/config/raw`
- Config update endpoint at `http://localhost:18081/config/update` (POST JSON: `{ "yaml": "..." }`)
- Restart endpoint at `http://localhost:18081/restart`

## Notes on Markets and Brokers

IBKR provides broad EU equity access including Borsa Italiana. Alpaca does not generally support EU stocks; use Alpaca for US markets or paper testing.
Trading only starts when at least one configured market is open (see `market.venues` and `market.open_mode`).
Holiday lists are empty by default; populate `market.venues[].holidays` per venue.

## Data Download (yfinance)

```bash
docker compose run --rm trader python3 -m app.main download --config /app/config/config.yaml --symbols AAPL MSFT
```

Data is saved to `/data` inside the container (mapped to `./data`).

If Yahoo blocks the container, set a proxy and slow down requests in `config/config.yaml`:

- `data.proxy` (e.g. `http://user:pass@proxy:8080`)
- `data.rate_limit_seconds` (e.g. `5`)

## Backtesting

```bash
docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml
```

## Learning (RL)

Train a PPO policy on local OHLCV data:

```bash
docker compose run --rm trader python3 -m app.main train --config /app/config/config.yaml
```

Enable learning in `config/config.yaml` by setting `learning.enabled: true`. A rule-based guardrail is configurable under `learning.guardrail`. When `learning.device` is set to `auto`, CUDA is used if available.
Training produces a report at `learning.training.report_path` with return, Sharpe, and drawdown metrics, plus charts in `learning.training.report_plot_dir`.
Models and reports are persisted under `./models` on the host.
Use `learning.training.resume: true` to reuse an existing model on restart, or set it to `false` to retrain from scratch.
The trainer writes a best-performing copy to `learning.best_model_path` when evaluation metrics improve; the trader loads it by default unless `learning.use_best_model: false`.

## Open Orders Awareness

The trader periodically fetches open orders and will skip new signals for symbols with pending orders (configurable via `execution.open_orders.*`). Pending buy orders are also treated as reserved cash for sizing.

## Pattern Trading Mode

Enable Pattern Trading by setting `strategy.name: pattern_trading` and configure:
- `pattern_trading.selection.*` for price/volume/gain/liquidity/catalyst filters
- `pattern_trading.pattern.*` for MA trend + pullback rules
- `pattern_trading.entry.*` for breakout and volume confirmation
- `pattern_trading.risk.*` for stop, partial take profit, and trailing stop

News catalysts use Alpaca’s news API by default (configure in `news.*`). If news data is unavailable, symbols are filtered out when `pattern_trading.selection.require_catalyst: true`.

Session gain is configurable via `data.session_gain_mode`:
- `gap`: compare current price to prior close
- `session`: compare to session open

## Dynamic Symbols (Alpaca Scanner)

Dynamic scanning is enabled by default to let the agent adapt the traded symbol list based on pattern filters:

```yaml
data:
  dynamic_symbols:
    enabled: true
    provider: alpaca
    feed: iex
    refresh_minutes: 5
    universe: alpaca_active
    cash_aware: true
    cash_buffer_pct: 95
    fallback:
      enabled: true
      relative_volume_min: 0.5
      premarket_gain_min_pct: 0.0
      min_shares_traded: 100000
      max_spread_pct: 2.0
      require_catalyst: false
```

The scanner applies `pattern_trading.selection.*` filters to Alpaca snapshots (price, volume, gain, spread, catalyst) and replaces the active symbols list when candidates are found.
When `cash_aware: true`, the scanner also caps the maximum price using current cash and `risk.max_position_size_pct` so the list adapts to low-fund scenarios.
If no candidates match the main filters, the optional `fallback` block relaxes filters to keep an affordable symbol list.

## Logging & Alerts

Log files are written to `logging.file_path` (default `/data/logs/trader.log`) with rotation settings under `logging.max_bytes` and `logging.backup_count`.
Prometheus alert rules live in `prometheus/alerts.yml`, and Grafana provisions an “Autotrader Alerts & Logs” dashboard for active alerts and error metrics.
Email alerting is configured via Alertmanager (`alertmanager/alertmanager.yml`) and runs on port `9093`.

## Multi-Strategy Mode

Set `strategy.names` to a list of strategies and choose how to combine them:
- `strategy.combine: priority` uses the first non-hold signal in the list order.
- `strategy.combine: vote` picks the majority action across strategies.

Available strategy names:
- `intraday_momentum`
- `pattern_trading`
- `rl_policy` (requires `learning.enabled: true`)

## AI Strategy Orchestrator

The AI orchestrator scores each strategy against the current market context and selects the top candidates per symbol.
Enable it under `orchestrator.*` in `config/config.yaml`:

```yaml
orchestrator:
  enabled: true
  mode: select
  top_k: 2
  min_score: 0.0
  learning:
    enabled: true
    learning_rate: 0.1
    min_bias: -1.0
    max_bias: 1.0
    decay: 0.02
    min_price_move_pct: 0.05
    state_path: /data/orchestrator_state.json
    save_interval_seconds: 300
    use_best_state: true
    best_state_path: /data/orchestrator_state_best.json
    best_score_path: /data/orchestrator_best_score.json
    score_ema_alpha: 0.1
```

Weights live under `orchestrator.strategy_weights` and include `momentum`, `trend`, `volatility`, `relative_volume`,
`session_gain_pct`, `spread`, and `catalyst`.
Learning updates a per-strategy bias based on the next price move after each decision and persists it in `state_path`.
The best-performing bias set is checkpointed to `best_state_path` and loaded by default when `use_best_state: true`.

Evaluate an existing model and regenerate charts:

```bash
docker compose run --rm trader python3 -m app.main evaluate --config /app/config/config.yaml
```

### Online Updates

Run online updates in a separate process:

```bash
docker compose run --rm learner
```

GPU online updates (requires NVIDIA Docker runtime):

```bash
docker compose --profile gpu up -d learner-gpu
```

To verify GPU visibility inside the container:

```bash
docker compose exec -T learner-gpu python3 - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

### Data Ingestion

Pull data from configured sources (`yfinance`, `stooq`, `alphavantage`):

```bash
docker compose run --rm trader python3 -m app.main ingest --config /app/config/config.yaml
```

### Config Reference (Learning + Data + Markets)

```yaml
market:
  open_mode: any
  venues:
    - name: BorsaItaliana
      timezone: Europe/Rome
      trading_hours:
        open: "09:00"
        close: "17:30"
      holidays: []
    - name: NYSE
      timezone: America/New_York
      trading_hours:
        open: "09:30"
        close: "16:00"
      holidays: []
    - name: Nasdaq
      timezone: America/New_York
      trading_hours:
        open: "09:30"
        close: "16:00"
      holidays: []

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
    resume: true
    report_path: "/app/models/training_report.json"
    report_plot_dir: "/app/models/reports"

data:
  output_dir: "/data"
  symbols: ["AAPL", "MSFT"]
  sources:
    - provider: yfinance
      enabled: true
      symbols: ["AAPL", "MSFT"]
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
docker compose run --rm trader python3 -m app.main trade --config /app/config/config.yaml
```

The trader iterates over every symbol listed in `data.symbols` each cycle. Trading is paused when all configured markets are closed.
Orders are sized based on available cash and the configured risk caps (`risk.max_position_size_pct`, `risk.max_short_exposure_pct`).

## Monitoring

Prometheus scrapes `trader:8001/metrics`.
Grafana auto-provisions a dashboard with:

- Trades total
- Trades rate by symbol/side
- Cumulative trades by symbol
- Active symbols (from config)
- Account equity, cash, invested
- Skipped orders by reason (e.g., insufficient cash, risk limits)
- Open orders by symbol/side
- Active broker (alpaca or ibkr)
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

## Git Repo Creation

Use the following steps to create a private repo and push:

```bash
git init
git add .
git commit -m "Initial Autotrader"

git remote add origin https://github.com/ilfrick/Autotrader.git

git push -u origin master
```

If you use the GitHub CLI:

```bash
gh repo create Autotrader --private --source . --remote origin --push
```

## Disclaimer

This software is for research/education. You are responsible for compliance with regulations, broker requirements, and all trading risk.

## Holiday Calendars

Per-venue holidays are stored under `market.venues[].holidays` in `config/config.yaml`. A `calendar-updater`
service refreshes these lists weekly (configurable) from online sources:

- NYSE: `https://www.nyse.com/markets/hours-calendars`
- Nasdaq: `https://www.nasdaqtrader.com/Trader.aspx?id=Calendar`
- Borsa Italiana: `https://date.nager.at` (Italy public holidays + Good Friday)

Update interval and horizon are controlled by:

```yaml
market:
  holiday_update:
    enabled: true
    interval_days: 7
    years_ahead: 1
```

One-off refresh:

```bash
docker compose run --rm calendar-updater python -m app.utils.holiday_update --config /app/config/config.yaml --once
```

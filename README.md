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

## Documentation

- `docs/README.md` for the full index.
- Subpages in `docs/` cover: AI filter, trading loop, strategies, execution, risk, brokers, backtesting, data, learning, API, monitoring, configuration.

## Architecture

```mermaid
flowchart LR
    subgraph Data["Market Data, Ingestion & Scanning"]
        YF[yfinance live data]
        AlpacaBars[Alpaca historical bars]
        AlpacaAssets[Alpaca assets/universe]
        Ingest[Ingest pipeline]
        Scan[Dynamic symbol scanner]
        AIFilter[AI symbol filter<br/>online updates + news]
        News[News catalyst fetcher]
    end
    subgraph Models["Model Store"]
        ModelStore[/data + /app/models/]
    end
    subgraph Core["Trading Loop"]
        Trader[TradingAgent]
        Strat[Strategies<br/>rl_policy / rl_policy_fees / intraday_momentum / pattern_trading]
        Orchestrator[Strategy Orchestrator<br/>ML + rules]
        Risk[Risk Manager]
        Exec[Execution Engine]
        Queue[Order Queue<br/>FIFO + feedback]
        Orders[Open order tracking<br/>cancel/skip]
    end
    subgraph Features["Feature Feeds"]
        AIFeatures[AI filter features<br/>filtered to active orders]
    end
    subgraph Training["Training & Pretrain"]
        RLTrain[RL training + online updates]
        OrchPretrain[Orchestrator pretrain]
        AIFilterTrain[AI filter training]
    end
    subgraph Backtest["Backtesting"]
        AgentBT[Agent backtest engine]
    end
    subgraph Broker["Broker Layer"]
        Alpaca[Alpaca]
        IBKR[IBKR]
    end
    subgraph Observability["Monitoring & Control"]
        Metrics[Prometheus Metrics]
        Grafana[Grafana Dashboard]
        Alerting[Alertmanager]
        API[FastAPI Config/UI]
        Logs[Rotating Logs]
    end

    YF --> Trader
    AlpacaBars --> RLTrain
    AlpacaBars --> AgentBT
    AlpacaBars --> AIFilterTrain
    AlpacaAssets --> Scan
    AlpacaAssets --> AIFilter
    Scan --> Trader
    AIFilter --> Trader
    AIFilter --> AIFeatures --> Orchestrator
    News --> Strat
    News --> AIFilter
    Ingest --> AlpacaBars
    RLTrain --> ModelStore
    OrchPretrain --> ModelStore
    AIFilterTrain --> ModelStore
    Trader --> Strat --> Orchestrator --> Risk --> Queue --> Orders --> Exec --> Alpaca
    Exec --> IBKR
    Alpaca --> Queue
    IBKR --> Queue
    Queue --> Orchestrator
    Orchestrator --> ModelStore
    AgentBT --> Trader
    Trader --> Metrics --> Grafana
    Metrics --> Alerting
    Trader --> Logs
    API <--> Trader
```

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

Backtesting defaults to `backtest.mode: agent`, which runs the actual trading logic (strategies + orchestrator +
risk + sizing) on CSV data. Set `backtest.mode: sma` to use the legacy SMA baseline.

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

Dynamic scanning is enabled by default to let the agent adapt the traded symbol list. Pattern strategy filters only
apply to the pattern strategy; other strategies use `data.dynamic_symbols.filters`.

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
    cash_cap_mode: cash
    cash_max_pct: 100
    fallback:
      enabled: true
      relative_volume_min: 0.0
      premarket_gain_min_pct: 0.0
      min_shares_traded: 100000
      max_spread_pct: 2.0
      require_catalyst: false
```

The scanner applies `pattern_trading.selection.*` filters only for the pattern strategy; other strategies use
`data.dynamic_symbols.filters` for price/volume/gain/spread/catalyst checks before replacing their symbol lists.
Maximum price is no longer capped by config; it is derived solely from available cash via `cash_aware`.
When `cash_aware: true`, the scanner caps the maximum price using available cash. Set `cash_cap_mode: risk`
to also respect `risk.max_position_size_pct`.
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
- `rl_policy_fees` (RL policy with fee-aware guardrail)

## AI Strategy Orchestrator

The RL orchestrator scores each strategy against the current market context (market state, strategy signals,
AI-filter-style features, and order feedback) and selects the top candidates per symbol.
Enable it under `orchestrator.*` in `config/config.yaml`:

```yaml
orchestrator:
  mode: select
  top_k: 2
  min_score: 0.0
  rl:
    enabled: true
    model_type: lstm
    device: auto
    model_path: /data/orchestrator_model.pt
    best_model_path: /data/orchestrator_model_best.pt
    use_best_model: true
    time_penalty_per_bar: 0.05
    pretrain:
      enabled: true
      in_trader: false
      lookback_days: 30
      interval: 5m
```

RL mode trains per-bar to maximize time-penalized returns using rewards based on the next price move. The default LSTM model uses
a rolling sequence of market features for each symbol. It maintains a replay buffer,
updates the model each bar, and checkpoints the best-performing model automatically. Pretraining pulls fresh
historical data via yfinance to warm start the policy.

Pretraining runs out-of-band by default (`orchestrator.rl.pretrain.in_trader: false`). To pretrain manually:

```bash
docker compose run --rm trader python3 -m app.main pretrain-orchestrator --config /app/config/config.yaml
```

For multi-year 5m pretraining, set the pretrain provider to Alpaca (yfinance intraday is capped at ~60 days):

```yaml
orchestrator:
  rl:
    pretrain:
      provider: alpaca
      alpaca_api_key: ${ALPACA_API_KEY}
      alpaca_api_secret: ${ALPACA_API_SECRET}
```

RL orchestrator sweep:

```bash
docker compose run --rm trader python3 scripts/orchestrator_sweep.py --config /app/config/config.yaml
```

To pretrain with randomly selected Alpaca symbols, set:

```yaml
orchestrator:
  rl:
    pretrain:
      symbols_source: alpaca_active_random
      interval: 5m
      window_days: 60
      coverage_days: 365
      step_days: 30
```

Rule-based orchestrator weights are no longer used; the RL orchestrator handles selection end-to-end.

## Fee-Aware RL Strategy

`rl_policy_fees` wraps the RL policy with a fee guardrail based on broker-specific commission settings. Configure
fees per broker and the guard thresholds:

```yaml
brokers:
  alpaca:
    fees:
      commission_pct: 0.0
      per_trade_fee: 0.0
      per_share_fee: 0.0
      min_fee: 0.0
      spread_pct: 0.0
strategy:
  fee_aware:
    min_edge_pct: 0.02
    edge_multiplier: 1.0
    min_notional: 50.0
```

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

Pull data from configured sources (`yfinance`, `alpaca`, `stooq`, `alphavantage`). Alpaca supports multi-year intraday bars.

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

## History

Recent changes (newest first):
- Added automatic Alpaca symbol-to-venue refresh for per-symbol market gating.
- Rebuilt and restarted the live stack after adding per-symbol venue gating.
- Added per-symbol venue gating to prevent orders when a symbol’s market is closed.
- RL orchestrator sweep running in dev (in progress).
- Orchestrator sweep rerun requested after image refresh.
- Orchestrator sweep now falls back to backtest CSV symbols when data.symbols is empty.
- Updated orchestrator sweep script to use RL orchestrator settings.
- Aligned orchestrator documentation with the RL implementation and marked legacy sweep usage.
- Enabled backtest.run_when_closed and restarted tests-when-closed.
- Rebuilt and restarted all services to apply the latest configuration.
- Rebuilt tests-when-closed to include backtest runner support.
- tests-when-closed now runs backtests when configured in backtest.run_when_closed.
- Ran pytest in the tests-when-closed container (10 passed).
- Dynamic backtest symbols now fall back to CSV data when symbols are empty.
- Added backtest plan sampling with local news support plus robustness/unit tests.
- Stopped the dev stack after publishing v2.0 release.
- Rebuilt and restarted the v2.0 stack after fixing trader startup crash.
- Fixed trader startup crash by initializing broker name before the order queue.
- Started a long-running v2.0 RL training run (in progress).
- Rebuilt the dev stack after enforcing GPU usage.
- Rebuilt and restarted the v2.0 stack after enforcing GPU usage.
- Enforced GPU usage for ML/RL components when CUDA is available.
- Started a long-running dev RL training run (in progress).
- Rebuilt the dev stack after RL training guard fix.
- Rebuilt and restarted the v2.0 stack after RL training guard fix.
- Guarded RL training against empty/short datasets to prevent index errors.
- Adjusted dev docker-compose ports to avoid conflicts and rebuilt the dev stack.
- Rebuilt and restarted the full v2.0 stack after orchestrator/backtest hardening.
- Hardened backtests against live data calls and disabled on-demand AI feature fetches in the orchestrator by default.
- Ignored dev worktree artifacts and cleaned up transient log/output files.
- Removed unused orchestrator config parameters now that the RL orchestrator is standard.
- Aligned AI/RL objectives with time-penalized return for faster equity growth.
- Updated architecture diagram to show AI filter features feeding the orchestrator.
- Enriched order feedback with broker status and fill metrics for the RL orchestrator.
- Updated architecture diagram to include the order queue and broker feedback loop.
- Added an order queue layer that serializes broker submissions and feeds order responses into the RL orchestrator.
- Replaced the orchestrator with an RL-based version that incorporates AI filter inputs for actionable symbols.
- Added documentation subpages per subsystem in `docs/`.
- Added pytest-based test suite and a market-closed test runner service.
- Fixed AI filter bar mapping for single-symbol responses; corrected backtest session prev-close handling.
- Updated learner lock logs to reflect idle behavior.
- Kept inactive learner in a sleep loop instead of exiting to avoid restart churn.
- Added learner lock heartbeats during training/sleep to prevent CPU/GPU contention.
- Enforced exclusive learner execution with GPU preference for online training.
- Added AI filter device logging for GPU/CPU confirmation.
- Enabled GPU acceleration for the AI symbol filter when CUDA is available.
- Fixed trader loop indentation regression causing container restarts.
- Updated architecture diagram to show AI filter ingesting news.
- Added news-aware features to the AI symbol filter.
- Synced news refresh to 1 minute to match AI filter cadence.
- Added logging for news catalyst cache refreshes.
- Increased AI filter online update steps and max symbols for continuous training.
- Updated architecture diagram to reflect AI filter, ingestion, and online updates.
- Enabled online updates for the AI symbol filter (incremental retraining on refresh).
- Increased AI filter cadence to 1 minute and raised universe cap for live scanning.
- Added a pre-run log for the AI filter so execution is visible immediately.
- Added AI filter heartbeat logging every 30s after a successful run.
- Added logging when the AI symbol filter runs so live usage is visible in logs.
- Ported dev run artifacts (backtest and ingest outputs) into v2.0 for traceability.
- Added Alpaca ingestion support to load a full universe when symbols are omitted; added saved backtest case configs.
- Promoted the AI dynamic symbol filter into the live branch for v2.0.
- Fixed open-order Prometheus gauges to remove stale labels so Grafana shows only current pending orders.
- Added strategy-level pending-order guard to skip signal evaluation while orders are open.
- Added pending-order cancel/replace logic and broker order cancellation support.
- Forced Grafana to reload provisioned dashboards for consistent axis autoscaling.
- Added AI-driven symbol scoring for full Alpaca US universe selection in dev.
- Added portfolio position metrics and Grafana table panels for holdings and pending orders.
- Ensured held positions are always evaluated and prevented short sells when no long position.
- Deployed combined RL strategies (`rl_policy` + `rl_policy_fees`) and isolated dynamic symbol lists per strategy.
- Added Alpaca historical ingestion for multi-year 5m data and wired ML orchestrator pretrain to use Alpaca data.
- Added agent-aligned backtest engine so the real trading loop (strategy + orchestrator + risk + execution) is tested.
- Added ML orchestrator (LSTM default) with online training, best-model checkpointing, and out-of-band pretraining.
- Added dynamic symbol scanning with cash-aware caps and fallback filters.
- Added web UI config editor and Grafana dashboards for active broker, strategies, orders, and account metrics.

Highest positive impact (testing/live trading):
- RL-only strategy in agent backtest: +25.76% return, 12 trades on the full-year run.
- Dual RL strategies with per-strategy symbol lists: +56.20% return in short-window dynamic-symbol tests.
- Best-model loading for RL (`learning.use_best_model: true`) keeps the highest-evaluated model in live runs.

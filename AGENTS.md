## Autotrader Agent Guide

This repo contains a Python intraday trading agent for US and EU equities (NYSE, Nasdaq, Borsa Italiana), with broker adapters, risk controls, backtesting, data download, and metrics/monitoring.

## Quick Orientation

- `app/main.py` is the CLI entrypoint with subcommands: `trade`, `backtest`, `download`, `api`, `train`, `online-train`, `evaluate`, `ingest`.
- Core loop: `app/agents/trader.py` + `app/strategies/intraday_momentum.py` + `app/execution/executor.py` + `app/risk/manager.py`.
- Broker adapters: `app/brokers/alpaca.py`, `app/brokers/ibkr.py`, abstract base in `app/brokers/base.py`.
- Backtesting: `app/backtest/agent_engine.py` runs the real `TradingAgent` loop on CSVs; legacy SMA lives in `app/backtest/engine.py`.
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

By default the `trader` service uses the GPU-enabled image. To force CPU execution for RL
inference/training, set `learning.device: cpu` in `config/config.yaml` (or via the web UI).
To disable GPU acceleration in backtests, set `backtest.use_gpu: false`.
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
docker compose run --rm trader python3 -m app.main download --config /app/config/config.yaml --symbols AAPL MSFT
```

- Pretrain ML orchestrator:

```bash
docker compose run --rm trader python3 -m app.main pretrain-orchestrator --config /app/config/config.yaml
```

- Sweep orchestrator hyperparameters:

```bash
docker compose run --rm trader python3 scripts/orchestrator_sweep.py --config /app/config/config.yaml
```

- Backtest:

```bash
docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml
```

- Ingest Alpaca multi-year bars (configure `data.sources` first):

```bash
docker compose run --rm trader python3 -m app.main ingest --config /app/config/config.yaml
```

- Trade:

```bash
docker compose run --rm trader python3 -m app.main trade --config /app/config/config.yaml
```

- Train RL policy (offline):

```bash
docker compose run --rm trader python3 -m app.main train --config /app/config/config.yaml
```

- Evaluate policy and regenerate charts:

```bash
docker compose run --rm trader python3 -m app.main evaluate --config /app/config/config.yaml
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
docker compose run --rm trader python3 -m app.main ingest --config /app/config/config.yaml
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
- Pattern Trading config lives under `pattern_trading` and is enabled via `strategy.name: pattern_trading`.
- Multi-strategy config uses `strategy.names` with `strategy.combine` set to `priority` or `vote` (enabled by default in `config/config.yaml`).
- AI strategy orchestration uses `orchestrator.*` to score strategies per symbol and select the top candidates each cycle. ML mode (`orchestrator.ml.enabled`) trains per bar with replay buffer + best-model checkpoints; default model is LSTM with `orchestrator.ml.seq_len`.
- Fee-aware RL is available as `rl_policy_fees`, using broker-specific fee config under `brokers.<name>.fees` plus guardrails in `strategy.fee_aware`.
- Orchestrator pretraining runs out-of-band by default (`orchestrator.ml.pretrain.in_trader: false`); use `python -m app.main pretrain-orchestrator` in Docker to warm-start the model.
- Orchestrator learning persists per-strategy bias updates under `orchestrator.learning.state_path`, and checkpoints the best biases to `orchestrator.learning.best_state_path` for default loading.
- Training writes a JSON report at `learning.training.report_path` and charts in `learning.training.report_plot_dir`.
- Models and reports are stored in `./models` via the Docker volume.
- `learning.training.resume` controls whether training resumes from an existing model or starts fresh.
- `learning.use_best_model` selects the best model copy (from `learning.best_model_path`) if available.
- Data ingestion sources are configured under `data.sources` (supports `yfinance`, `alpaca`, `stooq`, `alphavantage`).
- Dynamic scanner filters are configured under `data.dynamic_symbols.filters`, while `pattern_trading.selection` only
  affects the pattern strategy. Each strategy gets its own symbol list when dynamic scanning runs. Maximum price is
  derived from available cash (no config-based cap).
- News catalysts (for Pattern Trading) are configured under `news` (default Alpaca news API).
- Open-order tracking is configured under `execution.open_orders`.
- Alpaca keys come from `ALPACA_API_KEY` / `ALPACA_API_SECRET` in `.env`.
- `brokers.ibkr.enabled` controls IBKR adapter selection. If `false`, Alpaca is used.
- Data directory is `/data` inside containers (mapped to `./data` on host).
- `data.interval` and `data.lookback_days` are clamped for yfinance intraday limits.
- `data.session_gain_mode` controls session gain calculation (`gap` or `session`).
- Dynamic symbol scanning is configured under `data.dynamic_symbols` (Alpaca snapshot-based scanner), enabled by default, refreshes every 5 minutes, supports cash-aware filtering with `cash_aware` (see `cash_cap_mode`), and can relax filters via `data.dynamic_symbols.fallback`.
- `data.dynamic_symbols.universe: brokers_active` seeds the scanner/AI filter from enabled broker universes plus open positions and orders.
- `news.provider: brokers` aggregates catalysts across enabled brokers (Alpaca-backed today).
- Alerts are defined in `prometheus/alerts.yml` and a dedicated Grafana dashboard is provisioned for alerting/health.
- Alertmanager handles email notifications via `alertmanager/alertmanager.yml`.
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
- Dashboard also shows active symbols, active broker, and account equity/cash/invested from broker account data. Skipped orders are available via `orders_skipped_total` metrics.

## Extending the Codebase

- New strategies should extend `app/strategies/base.py` and be wired in `app/agents/trader.py`.
- New brokers should implement `app/brokers/base.py` and be added to `_build_broker` in `app/main.py`.
- Additional metrics belong in `app/monitoring/metrics.py`.

## Testing

No automated tests are present. For changes, prefer manual checks via:
- `python -m app.main backtest` in Docker.
- `/health` and `/config` endpoints via the `api` service.

## History

Recent changes (newest first):
- Cleared stale active-symbol metrics so Grafana only shows current symbols.
- Ensured held positions stay in dynamic symbols even when scanner filters exclude them.
- Raised minimum trade price to 2.0 across dynamic scanning and pattern selection.
- Added universe price filtering by cash-aware price bounds for dynamic symbols.
- Added Grafana panel for broker API call activity.
- Fixed Mermaid label text so the architecture diagram renders in master.
- Ensured dynamic universe always keeps positions/orders and hardened broker-backed news fetching.
- Fixed architecture diagram to show broker-backed news inputs.
- Fixed OrderQueue snapshot response handling so tests pass.
- Updated the architecture diagram to show broker-backed news and broker universe inputs.
- Added broker-backed news catalysts and a broker-aware universe option for the AI symbol filter.
- Fixed multi-broker symbol aggregation, action-based routing, and fallback routing.
- Guarded routing default to only select enabled brokers.
- Added multi-broker routing support with broker-aware metrics and alerts.
- Added rejection reason labels to order rejection alerts and logs.
- Added order rejection metrics/alerts with broker and error code.
- Guarded sell actions to skip when no long position exists.
- Added config key validation in the web UI update flow to block typos.
- Seeded symbols from checkpoint so active symbols persist during AI filter startup.
- Initialized dynamic symbol cache to prevent checkpoint crashes after async refresh.
- Made AI filter refresh async so the trader keeps the last valid symbols during updates.
- Reduced AI filter lookback_days to 2 to speed live scoring.
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
- Added Alpaca historical ingestion for multi-year intraday data and ML pretrain support.
- Added agent-aligned backtest engine for realistic strategy/orchestrator/risk testing.
- Introduced ML orchestrator (LSTM default) with online training and best-model checkpoints.
- Added dynamic symbol scanning with cash-aware caps and fallback filters.
- Added web UI config editor and Grafana dashboards for broker/strategy/account visibility.

Highest positive impact (testing/live trading):
- RL-only backtest returned +25.76% with 12 trades on the full-year run.
- Dual RL strategies with per-strategy symbol lists returned +56.20% on short-window dynamic-symbol tests.
- Best-model loading keeps the strongest evaluated RL policy in live trading.

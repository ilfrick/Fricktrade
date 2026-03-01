# Scripts

Operational and development scripts for Fricktrade.

## Startup & Deployment

| Script | Description |
|--------|-------------|
| `compose_up.sh` | Start all services with GPU auto-detection; generates per-account Grafana dashboards from `.env` before startup |
| `enable_gpu.sh` | Re-enable GPU after it was disabled by OOM handler; sets `data/gpu_state.json` to `gpu_disabled: false`, requires restart |
| `render_alertmanager_config.py` | Template-expand `alertmanager/alertmanager.yml` with `${VAR}` placeholders from `.env` and OS environment; writes `alertmanager.generated.yml` |
| `generate_grafana_dashboards.py` | Generate per-account Grafana dashboard JSON from template; reads `ALPACA_ACCOUNT_NAMES`/`IBKR_ACCOUNT_NAMES` from `.env` |

## Data & Backtesting

| Script | Description |
|--------|-------------|
| `download_data.sh` | Download OHLCV data for hardcoded symbols (AAPL, GOOGL, TSLA, MSFT) via Docker |
| `backtest_gpu.sh` | Run GPU-accelerated backtests via `docker compose --profile gpu` |
| `benchmark_runner.py` | Multi-scenario backtest benchmarks with Sharpe/Sortino/Calmar ratios, bootstrap CIs, and PDF report |
| `orchestrator_sweep.py` | Hyperparameter grid search for RL orchestrator (model type, hidden dim, seq len, learning rate) |
| `top_movers_forecast.py` | Build a walk-forward model from historical `*_YYYY-MM-DD_1m.csv` data and forecast next-day top movers |
| `top_movers_same_day.py` | Same-day models: early-session top-mover nowcast plus intraday low-zone entry scoring (`--model random_forest|xgboost`) |
| `tune_xgb_same_day.py` | Random-search tune XGBoost for same-day nowcast/entry objectives and emit best-params final run |

## Monitoring

| Script | Description |
|--------|-------------|
| `monitor_agent_session.sh` | Capture decision traces, Docker logs, and metrics during a trading session (default: Europe/Rome 15:30-22:00) |
| `monitor_full_session.sh` | Full US session monitor: decision traces, order flow, strategy signals, risk blocks, position changes, PnL snapshots, and end-of-session summary report |
| `monitor_cache_health.sh` | Snapshot health check: container status, market-cache logs, Redis ping; appends to `monitoring_checks.log` |
| `monitor_cache_latency.py` | Track market cache bar freshness, stale data ages, and container failures until NYSE close |
| `monitor_sell_decisions.py` | Parse decision traces for sell/exit signals and skip reasons; generate JSON report with RL value estimates and orchestrator picks |
| `monitor_sell_analysis.sh` | Capture exit triggers, success/failure rates, retry budget usage, pending notional, and order rates by side; schedule via cron at 15:25 CET Mon-Fri |
| `decision_monitor.py` | Incremental decision trace aggregator: byte-offset reader, P0 fix verification (weights, phantom sells, sell overshoot), periodic JSONL snapshots, session report; schedule via cron at 15:25 CET Mon-Fri |
| `analyze_top_movers_trade_gap.py` | Diagnose why reported top movers were not traded by streaming decision traces and classifying symbol-level blockers |
| `live_pnl_summary.sh` | Query Prometheus for live PnL, drawdown, trades, win rate, leverage, and risk metrics |

## Return Ranker

| Script | Description |
|--------|-------------|
| `collect_return_ranker_data.py` | Collect daily return-ranker training data (market features, news, next-day returns) from decision traces and Alpaca; triggers rolling-window retraining (`max_age_days=3`) |
| `collect_crypto_training_data.py` | Hourly crypto return-ranker training data collector (no market-hours gate); fetches 5m bars from Alpaca for crypto symbols derived from the live universe (`alpaca_active_all`), labels with 60-min forward returns, appends to `data/training/crypto_ranker_{date}.csv`; triggers retraining after 100+ rows. Cron: `0 * * * *` |
| `reset_checkpoints.sh` | Clear all model checkpoints and training state; use before a clean retrain from scratch |

## LLM Analysis

| Script | Description |
|--------|-------------|
| `post_session_analyst.py` | Post-session LLM analyst: reads decision monitor JSONL + trace, calls Gemini to produce a structured JSON report in `data/session_reports/report_YYYY-MM-DD.json`; schedule via cron `30 22 * * 1-5` (22:30 CET) |

## Profiling & Testing

| Script | Description |
|--------|-------------|
| `profile_live.sh` | Profile live trader with cProfile + py-spy flamegraph during market hours (runs inside Docker) |
| `run_live_profile.sh` | Host wrapper: stops trader, rebuilds image, runs `profile_live.sh` in container with SYS_PTRACE, restarts services |
| `run_tests_when_closed.py` | Periodic loop running pytest, backtests, and benchmarks when market is closed; respects ops_state |

## Usage Examples

```bash
# Start the stack
./scripts/compose_up.sh

# Monitor a live session
./scripts/monitor_agent_session.sh

# Run benchmarks
python3 scripts/benchmark_runner.py --config config/config.yaml --pdf-path data/reports/benchmark.pdf

# Monitor sell decisions
python3 scripts/monitor_sell_decisions.py --start "2026-02-11 15:30" --tz Europe/Rome

# Monitor cache latency
python3 scripts/monitor_cache_latency.py --config config/config.yaml --interval 60

# Monitor decision traces (aggregated session analysis)
python3 scripts/decision_monitor.py --trace-dir data/reports/decision_trace --output-dir data/monitoring

# Profile live trader
./scripts/run_live_profile.sh

# Orchestrator hyperparameter sweep
python3 scripts/orchestrator_sweep.py --config config/config.yaml --runs 12
```

## Usage Examples (continued)

```bash
# Run post-session LLM analysis (after market close)
python3 scripts/post_session_analyst.py --date 2026-02-28 --backend gemini

# Collect return-ranker training data
python3 scripts/collect_return_ranker_data.py --date 2026-02-28

# Reset all model checkpoints
./scripts/reset_checkpoints.sh
```

## Environment Variables

Most scripts read from `.env` in the project root. Key variables:

- `ALPACA_API_KEY`, `ALPACA_API_SECRET` — Broker credentials
- `ALPACA_ACCOUNT_NAMES` — Comma-separated account names (used by dashboard generation)
- `IBKR_ACCOUNT_NAMES` — IBKR account names (optional)
- `GRAFANA_ADMIN_PASSWORD` — Grafana password from env
- `ANTHROPIC_API_KEY` — Claude API key (required for LLM sentiment and risk interpreter)
- `GOOGLE_GEMINI_API_KEY` — Gemini API key (required for LLM post-session analyst and symbols filter)

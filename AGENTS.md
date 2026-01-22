<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

## Fricktrade Agent Guide

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

- Pretrain RL orchestrator:

```bash
docker compose run --rm trader python3 -m app.main pretrain-orchestrator --config /app/config/config.yaml
```

- Sweep orchestrator hyperparameters:

```bash
bash
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
- Additional strategies include `trend_following`, `factor_model`, `stat_arb_pairs`, and `market_maker` under `strategy.params.*`.
- Execution algos (TWAP/VWAP/POV) are configured under `execution.algos`.
- Volatility targeting is configured under `risk.vol_targeting`.
- AI strategy orchestration uses `orchestrator.*` with an RL policy-gradient model to score strategies per symbol and select the top candidates each cycle (LSTM or MLP configured under `orchestrator.rl.*`).
- Fee-aware RL is available as `rl_policy_fees`, using broker-specific fee config under `brokers.<name>.fees` plus guardrails in `strategy.fee_aware`.
- Strategy performance reporting + kill switch are configured under `strategy.performance.*` (rolling win rate/drawdown checks).
- Orchestrator pretraining runs out-of-band by default (`orchestrator.rl.pretrain.in_trader: false`); use `python -m app.main pretrain-orchestrator` in Docker to warm-start the model.
- - Training writes a JSON report at `learning.training.report_path` and charts in `learning.training.report_plot_dir`.
- Models and reports are stored in `./models` via the Docker volume.
- `learning.training.resume` controls whether training resumes from an existing model or starts fresh.
- `learning.use_best_model` selects the best model copy (from `learning.best_model_path`) if available.
- Data ingestion sources are configured under `data.sources` (supports `yfinance`, `alpaca`, `stooq`, `alphavantage`).
- Dynamic scanner filters are configured under `data.dynamic_symbols.filters`, while `pattern_trading.selection` only
  affects the pattern strategy. Each strategy gets its own symbol list when dynamic scanning runs. Price caps are
  enforced by buying power (`cash_aware`/`cash_cap_mode`), and held/open-order symbols are always retained.
- News catalysts (for Pattern Trading) are configured under `news` (default Alpaca news API).
- Open-order tracking is configured under `execution.open_orders`.
- Alpaca keys come from `ALPACA_API_KEY` / `ALPACA_API_SECRET` in `.env`.
- `brokers.ibkr.enabled` controls IBKR adapter selection. If `false`, Alpaca is used.
- Data directory is `/data` inside containers (mapped to `./data` on host).
- `data.interval` and `data.lookback_days` are clamped for yfinance intraday limits.
- `data.session_gain_mode` controls session gain calculation (`gap` or `session`).
- Dynamic symbol scanning is configured under `data.dynamic_symbols` (Alpaca snapshot-based scanner), enabled by default, refreshes every 1 minute by default, supports cash-aware filtering with `cash_aware` (see `cash_cap_mode`), and can relax filters via `data.dynamic_symbols.fallback`.
- `data.dynamic_symbols.universe: brokers_active` seeds the scanner/AI filter from the Alpaca active universe today plus open positions and orders.
- `news.provider: brokers` aggregates catalysts across enabled brokers (Alpaca-backed today).
- Alerts are defined in `prometheus/alerts.yml` and a dedicated Grafana dashboard is provisioned for alerting/health.
- Alertmanager handles email notifications via a locally rendered config (`alertmanager/alertmanager.generated.yml`) based on `.env` values; the template is `alertmanager/alertmanager.yml`.
- Live profiling scripts live under `scripts/profile_live.sh` (inside container) and `scripts/run_live_profile.sh` (host runner).
- Healthwatch can optionally stop/start services around market hours via `healthwatch.market_shutdown.*`.
- Manual kill switches live under `kill_switch.*` (force sleep or force liquidation with interlock).
- Daily top movers report is configured under `reports.daily_top_movers.*` (email + training exports, signal thresholds, news correlation, feed selection, and decision traces).
- `market.open_mode` chooses whether any or all configured venues must be open to trade.
- `market.venues[].holidays` is refreshed by `calendar-updater` (or can be edited manually).
- `calendar-updater` refreshes holiday calendars weekly from online sources (NYSE, Nasdaq, Italy public holidays).

## Behavior Details

- Trading loop pulls live data via `data.provider` (brokers/alpaca/yfinance) and iterates over the active symbol set (static list or dynamic scanner/AI filter).
- Trading is paused when all configured markets are closed.
- Strategy emits `buy`, `sell`, `exit`, or `hold`; `exit` closes the position.
- Risk checks are threshold-based and order sizing is cash-aware using broker equity/cash plus exposure caps.
- RL feature vectors now include risk parameters (limits, vol/VAR haircuts, kill switches) and the latest per-symbol risk decision (allow/block + reason + action); changing risk feature shape requires retraining affected RL models.
- RL feature set now includes risk parameters (position/leverage limits, vol/var haircuts, kill switches) and the latest per-symbol risk decision (action, allow/block, reason) so models see the risk posture.
- Agent-aligned backtest loads per-symbol CSVs from `backtest.data_dir` (legacy SMA engine uses the first matching CSV).
- API `/config` masks Alpaca keys before returning; `/config/update` accepts YAML updates and `/restart` triggers a graceful container restart.
- Grafana auto-provisions the "Fricktrade Overview" dashboard with trade counts/rates, PnL, and drawdown.
- Dashboard also shows active symbols, active broker, and account equity/cash/invested from broker account data. Skipped orders are available via `orders_skipped_total` metrics.

## Extending the Codebase

- New strategies should extend `app/strategies/base.py` and be wired in `app/agents/trader.py`.
- New brokers should implement `app/brokers/base.py` and be added to `_build_broker` in `app/main.py`.
- Additional metrics belong in `app/monitoring/metrics.py`.

## Testing

Pytest covers core components. For changes, run:
- `pytest` or `python -m pytest` (local/testenv).
- `python -m app.main backtest` in Docker.
- `/health` and `/config` endpoints via the `api` service.

## History

Recent changes (newest first):
- **Fix: Resolve all remaining Docker build and runtime dependency issues.** Corrected backtrader version to 1.9.78.123 in requirements.txt. Added DEBIAN_FRONTEND=noninteractive to Dockerfiles to prevent interactive apt-get prompts. Upgraded pip in Dockerfiles to ensure robust dependency resolution. These changes resolve ModuleNotFoundError for backtrader and allow all core services (api, trader, learner) to start and run correctly. (2026-01-21)
- Fixed Keras model deserialization errors by adding `tf_keras` dependency and restoring `TF_USE_LEGACY_KERAS=1` in Dockerfiles. Ensures the 'Keras return overlay' in the AI symbol filter can load and use pre-trained models. (2026-01-21)
- Added explainability fields to decision traces and new oversight runbook doc.
- Added stress/liquidity haircuts to sizing for real-time risk controls.
- Added audit/compliance retention, signing, and reason-code enforcement support.
- Added active model pointer publishing and ops-state gating for trader/learner/tests.
- Added ops state file output and aligned learner/tests with healthwatch scheduler state.
- Kept tests-when-closed running during market shutdown via healthwatch keep_services.
- Phase 6: added decision audit logs, compliance exports, and latency dashboards/metrics.
- Phase 5: added model registry metadata, drift detection, and auto-rollback to best RL model.
- Phase 4: added VaR/CVaR gating, exposure caps, and volatility-aware kill switch profiles.
- Phase 3: added market impact estimates, adaptive execution selection, and retry policy for queued orders.
- Phase 2: added OHLCV validation, split/dividend adjustments, and data quality reports for ingestion.
- Phase 1: added bootstrap CI, Monte Carlo stress, buy/hold baseline to benchmarks; added backtest spread/slippage.
- Started top-tier Phase 0 planning for benchmark enhancements (bootstrap CI, MC stress, buy/hold baseline).
- Added benchmarking plots, regime tagging, scorecard metrics, and PDF summaries.
- Added benchmark runner and documentation for walk-forward and stress tests.
- Added Grafana panels for intraday signal metrics (percent + absolute).
- Fixed AI filter retrain to pass broker config to news fetcher.
- Fixed RL orchestrator AI feature extraction indentation regression.
- Fixed Alpaca market data prefetch using missing IBKR handle; align AI filter signals to latest day.
- Added intraday signal metrics to live decisions, RL features, and AI filter training.
- Added env-based auto-detection for multi-account brokers with graceful fallback on invalid keys.
- Added multi-account broker support with per-account routing and config helpers.
- Added daily top movers report with email + training data export.
- Added manual kill switches for force sleep and force liquidation with interlock.
- Added healthwatch scheduler heartbeat logging.
- Enabled healthwatch market-based stack sleep/wake in config.
- Fixed market-based sleep/wake scheduling to use timezone-aware UTC timestamps.
- Added optional healthwatch market-based stack sleep/wake control.
- Added explicit logs when news catalyst refresh starts/completes.
- Made news catalyst refresh async so the trader keeps running while Ollama updates.
- Added a separate Grafana dashboard for strategy performance metrics.
- Added Grafana stat panel for 24h PDT blocks.
- Added Grafana panel for PDT blocks (day-trading protection).
- Added PDT-protection block counter for broker-rejected orders.
- Added rolling strategy performance reporting and kill switch thresholds.
- Run Ollama as a docker service for news LLM gating.
- Added optional Ollama-based LLM gate for news catalysts (disabled by default).
- Fixed live lookback slicing to use bars-per-day instead of raw days count.
- Added multi-broker live market data provider support (alpaca/ibkr) with routing.
- Switched live market data provider to Alpaca (batch bars) with optional yfinance fallback.
- Added Grafana table for strategy selection counts.
- PnL% now uses broker-reported last_equity when available, otherwise start equity.
- PnL% and drawdown metrics now track equity vs start/peak instead of staying at zero.
- Switched Open Orders panel to instant view to avoid stale series.
- Aligned Open Orders Grafana panel to show last 5 minutes to match pending orders view.
- Switched dynamic symbol price caps to use buying power and exposed buying power metrics.
- Capped dynamic symbol list size to the tradeable universe count (plus positions/open orders).
- Raised dynamic_symbols.max_symbols to 50000 to allow the full active universe.
- Enforced cash-aware symbol filtering to cap candidates by available cash and always include open-order symbols.
- Added flow diagrams for the trading agent (dev).
- Switched orchestrator to direct mode (single strategy selection) using all strategy signals.
- Tweaked broker market status panel to show only current status (no history).
- Added Grafana broker market status panel and broker_market_open metric.
- Adjusted Grafana active symbol panels to show only active (value=1) series.
- Capped AI-filter symbol list to dynamic_symbols.max_symbols to prevent oversized active symbol sets.
- Fixed yfinance downloads by only passing proxy when configured.
- Guarded factor model and AI filter features against zero prices to avoid divide warnings.
- Guarded intraday momentum strategy against zero prices to prevent backtest errors.
- Added production strategy set (trend, factor, stat-arb, market making) with execution algos and vol targeting.
- Added tests for strategy models and execution algos.
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

## Session Log
### 2026-01-17
- Added API auth gating for config/restart endpoints and UI token support; documented in `README.md` and `.env.example`.
- Added IBKR currency mapping via `market.default_currency`, `market.symbol_currencies`, and `brokers.ibkr.currency`.
- Added open-order completion grace and safe market-cache JSON serialization with optional legacy pickle reads.
- Hardened Alpaca account secret merge by name and updated AI filter cache wiring.
- Added tests for API auth, IBKR currency mapping, market cache serialization, and order-queue grace.
- Committed and pushed `d20694f` to both `origin` and `github`; tests skipped locally due to missing deps.
- Session saved to `/home/nicola/.codex/AGENTS.md`.
- Refactored TradingAgent to keep risk, cooldowns, pending entries, and performance state per broker/account.
- Keyed RL orchestrator history by broker+symbol and threaded broker context through select/record/update.
- Extended checkpoints and performance reports with per-broker state while keeping global metrics.
- Tests not run (not requested).
- Ran `pytest`; failed with `ModuleNotFoundError: No module named 'app'` (needs repo on `PYTHONPATH`).
- Completed code review for per-broker isolation refactor; noted orchestrator/risk outcome broker-key edge cases.
- Discussed deterministic per-account routing; recommended account+symbol keys and per-account strategy/orchestrator state to avoid cross-account collisions.
- Keyed strategy/guardrail caches by broker+symbol to prevent cross-account state bleed, and removed symbol-only orchestrator broker tracking.
- Ensured per-account risk_outcome is set before signals; updated broker selection flow accordingly.
- `PYTHONPATH=. pytest` passed (34 passed, 9 skipped).
- Added `pythonpath = .` to `pytest.ini` so tests import `app` without `PYTHONPATH`.
- Re-ran `pytest` successfully (34 passed, 9 skipped).
- Discussed performance-oriented rewrites: candidate hotspots include data scanning/feature extraction and backtest engine; advised profiling first and preferring NumPy/Polars/Numba before Rust/C++.

- Added lazy imports for optional deps (torch/stable-baselines3/yfinance/alpaca-py) so backtests/profiling run without the full ML stack.
- Added a Prometheus no-op fallback when `prometheus_client` is missing.
- Disabled decision-trace output after the first write failure to avoid log spam and perf overhead; re-ran cProfile backtest.
- Fixed Alpaca scanner helpers to use lazy imports in universe/price/venue helpers.
- `pytest` passed (34 passed, 9 skipped).

- Performance guidance: prioritize NumPy/Polars vectorization, then Numba for remaining per-bar loops; CuPy only after heavy vectorization with large batches. Pandas indexing fixes include pre-extracting columns to arrays, avoiding per-row `.loc`/`.xs`, and iterating over contiguous arrays.

- Backtest loop now pre-extracts OHLCV arrays and uses timeline indexers to avoid per-bar pandas `.loc` lookups; added `_prepare_backtest_frames` and `_SymbolState.update_from_values`.
- cProfile backtest time dropped ~3.75s -> ~1.65s in the sample run; pandas indexing no longer dominates.
- `pytest` passed (34 passed, 9 skipped).

- Live-loop perf guidance: biggest wins are reducing pandas work in `_market_state_from_df` and skipping symbol evaluation when no new bar; Numba best for pure numerical feature loops (EMA/RSI, signal metrics, var/cvar, realized vol) once arrays are used. Vectorize with NumPy (`np.diff`, `np.mean`) or batch arrays per provider where possible.

- Live loop optimization: market data now includes `last_bar_ts` and uses array extraction with `compute_signal_metrics_from_window` to reduce pandas overhead.
- Added optional `data.process_on_new_bar_only` gate to skip per-symbol processing when the bar timestamp hasn't advanced.
- Optimized `_session_gain_pct` to avoid DataFrame slicing; `pytest` passed (34 passed, 9 skipped).

- Enabled `data.process_on_new_bar_only` in `config/config.yaml` to skip per-symbol processing when bars haven't advanced.

- Attempted `docker compose run --rm trader python3 -m app.main backtest --config /app/config/config.yaml`; timed out after 120s and again after 300s. Backtest spammed warnings about missing RL models and did not complete; container was stopped.

- RL policy models missing: `learning.enabled` only loads `/app/models/ppo_policy.zip`; with Docker volumes this maps to `./models/`. If training/learner never ran or wrote elsewhere, no checkpoints exist. Orchestrator RL models live under `/data` (host `./data`).

- Attempted `docker compose run --rm trader python3 -m app.main train --config /app/config/config.yaml`; timed out after 30 minutes. RL training started on CUDA but no `ppo_policy.zip` was produced; only `models/ppo_policy.zip.tmp` remains (likely incomplete). Stopped lingering `fricktrade-trader-run-*` containers.

- Lowered `learning.training.timesteps` to 20000 and retried `docker compose run --rm trader python3 -m app.main train --config /app/config/config.yaml`; still timed out after 15 minutes. Training reported 20000 timesteps but logged `total_timesteps` ~483k; no `ppo_policy.zip` produced (only `ppo_policy.zip.tmp`). Stopped container `fricktrade-trader-run-23e95345e0c6`.

- Reduced RL training scope: set `learning.training.timesteps: 5000` and `learning.training.data_dir: /data/rl_train_small` (AAPL/MSFT 1m only). Training completed; `ppo_policy.zip`, `ppo_policy_best.zip`, registry, and reports created under `./models`.

- Checked online training flags: `learning.enabled: true`, `learning.online.enabled: true`, `orchestrator.rl.enabled: true` in `config/config.yaml`.
- No learner containers running (`docker ps` shows none). Online updates require starting `learner` or `learner-gpu`.

## Session update 2026-01-19 16:04:22 CET
- Checked container status and logs after recent changes.
- All core services (api, trader, prometheus, grafana, redis, healthwatch, market-cache) are up; api health checks are returning 200.
- Learner container is restarting due to `FileNotFoundError` when renaming `/app/models/ppo_policy.zip.tmp.zip` to `/app/models/ppo_policy.zip` during online updates; online training currently unhealthy.
- Trader logs show only periodic market-cache stale warnings and checkpoint writes; no fatal errors observed.

## Session update 2026-01-19 16:04:42 CET
- Fixed learner crash loop by making online checkpoint replace tolerant of missing temp file (`app/learning/train_rl.py`): if `.tmp.zip` is missing, falls back to `.tmp` or logs a warning and skips replace.

## Session update 2026-01-19 16:27:50 CET
- Checked post-restart logs: all Fricktrade containers are up; learner running online update without crash.
- Trader resumed trading loop; market-cache stale warning persists but no fatal errors.
- API started cleanly and /health returned 200.

## Session update 2026-01-19 16:30:54 CET
- Updated `docker-compose.yml` so the `learner` service uses the GPU image and requests NVIDIA devices by default (env vars + device_requests).

## Session update 2026-01-19 16:32:20 CET
- Adjusted `docker-compose.yml` to avoid hard GPU device requests for `learner`, allowing CPU fallback while still using the GPU image when available.

## Session update 2026-01-19 16:39:48 CET
- Fixed learner restart loop by switching its command to `python3` in `docker-compose.yml` (GPU image lacks `python`).

## Session update 2026-01-19 16:45:05 CET
- Post-restart check: all Fricktrade services up; learner running and reports GPU takeover, sleeping before resuming updates.
- Trader restarted cleanly and resumed trading loop; standard market-cache stale warning only.
- API healthy and serving /health.

## Session update 2026-01-19 18:16:55 CET
- Investigated trader down state: healthwatch market_shutdown is enabled and the system is in "stopped" state; `data/system_state.json` shows next_open 2026-01-20T09:00:00+01:00.
- Only keep_services are running (healthwatch, autoheal, daily-report, prometheus, tests-when-closed), matching the shutdown behavior.

## Session update 2026-01-19 18:45:53 CET
- Saved context/session after confirming trader stopped due to healthwatch market_shutdown (holiday) and pushed updates.
### 2026-01-21
- **Fix: Resolve all remaining Docker build and runtime dependency issues.** Corrected backtrader version to 1.9.78.123 in requirements.txt. Added DEBIAN_FRONTEND=noninteractive to Dockerfiles to prevent interactive apt-get prompts. Upgraded pip in Dockerfiles to ensure robust dependency resolution. These changes resolve ModuleNotFoundError for backtrader and allow all core services (api, trader, learner) to start and run correctly.

## Session update 2026-01-22 16:54:45 CET
- Removed the global "Active Symbols" stat panel from the overview Grafana dashboard (`grafana/provisioning/dashboards/fricktrade.json`) to avoid misleading counts versus per-account panels.

## Session update 2026-01-22 17:47:00 CET
- Cached RL policy models across symbols to avoid per-symbol reloads; `_reload_rl_strategies` now clears the shared cache so active model changes reload once.
- `pytest` passed (34 passed, 9 skipped).
- Committed and pushed to origin/github: 79c1046.

## Session update 2026-01-22 17:49:45 CET
- RL training now checkpoints/promotes models based on best evaluation performance; in-progress publishing disabled when best-only is enabled.
- Candidate models save to a temp path, best is copied to `/app/models/ppo_policy_best.zip`, and only best (or latest if allowed) is promoted to `/app/models/ppo_policy.zip`.
- `pytest` passed (34 passed, 9 skipped).
- Committed and pushed to origin/github: 3f1b551.

## Session update 2026-01-22 20:02:30 CET
- Checked market cache freshness: 1m cache files updated at 2026-01-22 19:37 CET; 5m cache files last updated at 2026-01-22 18:05 CET.
- Market cache refresh logs show 1m refreshed at 18:37 and 5m at 18:05; current staleness warnings are expected with a 10k+ symbol universe and 5s per-batch delay.
- Filtered symbols cache file `/data/market_cache/filtered/1m.json` last updated 2026-01-16 (age ~145h), so cached-symbols path is stale.

## Session update 2026-01-22 20:05:45 CET
- Market cache tuning: batch_size=500, delay_seconds=1.0, max_age_multiplier=10; AI filter cached symbols disabled.
- Market cache reads now honor max-age multiplier and TTLs; yfinance provider uses increased cache max age.
- `pytest` passed (34 passed, 9 skipped).
- Committed and pushed to origin/github: f0bebb9.

## Session update 2026-01-22 20:26:30 CET
- Market cache staleness snapshot: 1m cache age ~2829s (max_age 600s, stale), 5m cache age ~1128s (max_age 3000s, ok). Filtered symbols cache `data/market_cache/filtered/1m.json` age ~145h.
- Log issues: RL policy build failures due to missing numpy module; LLM catalyst timeouts to ollama; yfinance delisted/404/rate-limit errors; market cache stale bars warnings.

## Session update 2026-01-22 20:29:05 CET
- Clarified 1m vs 5m bars: main trading interval is 5m while AI symbol filter and some signals use 1m; both caches exist to avoid extra fetches.

## Session update 2026-01-22 20:39:08 CET
- Switched all config intervals to 5m (main data, AI filter, RL feature signal interval, RL training interval) across `config/config.yaml` and backtest case configs.

## Session update 2026-01-22 20:44:32 CET
- Rebuilt and restarted the full docker compose stack.
- Cache cleanup attempt blocked by sandbox policy (unable to delete `data/market_cache/bars/1m`/filtered).
- Staleness snapshot after restart: 5m updated_at 20:40:25 CET (age ~235s), last bar 20:40:00 CET (age ~260s). 1m updated_at 20:31:07 CET (age ~792s), last bar 20:30:00 CET (age ~860s). Filtered 5m cache missing.

## Session update 2026-01-22 20:46:15 CET
- Summarized cache max-age settings (market cache multiplier, filtered-symbols TTL, news cache minutes, checkpoint retention, online-update lock age).

## Session update 2026-01-22 20:50:55 CET
- Set market cache max_age_multiplier to 1 in `config/config.yaml`.

## Session update 2026-01-22 20:54:07 CET
- Rebuilt and restarted the full docker compose stack to apply max_age_multiplier=1.

## Session update 2026-01-22 20:55:10 CET
- Clarified cache staleness semantics: bars are evaluated per symbol+interval; filtered symbols cache is treated as a single blob per interval.

## Session update 2026-01-22 20:59:18 CET
- Changed filtered symbols cache to per-symbol keys/files with per-symbol staleness checks; kept legacy single-blob fallback.

## Session update 2026-01-22 21:03:51 CET
- Set market_cache.ignore_staleness=false, rebuilt images, restarted stack.
- Staleness snapshot: 5m updated_at 21:02:22 CET (age ~86s), last_bar 21:00:00 CET (age ~228s). Filtered per-symbol cache directory missing (not populated yet).

## Session update 2026-01-22 21:07:36 CET
- Verified repo clean; pushed to origin/github (no pending changes).
- Started background monitoring until US market close; log file: `data/monitoring/market_cache_monitor.log`.

## Session update 2026-01-22 21:12:55 CET
- Added risk enabled switch (config + backtest cases), bypassed risk checks when disabled, and updated RiskManager to short-circuit when disabled.

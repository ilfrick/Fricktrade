# Autotrader Project Context

This file captures the key state and workflows for this repo so a new Codex session
can reload context quickly. Keep it updated when the setup changes.

## Worktrees and Branches
- Live worktree: `/path/to/Autotrader` on branch `v1.0`
- Dev worktree: `/path/to/Autotrader/dev` on branch `master` (created via `git worktree`)

## Dev vs Live Intent
- Live (v1.0) is running and trading.
- Dev is for development and backtesting only; no live broker credentials.
- `dev/.env` exists and is based on `.env.example` with empty broker keys.

## Docker Notes
- Main compose file: `docker-compose.yml`
- Ports are fixed in compose; to run dev alongside live, use a separate project name:
  - Example: `docker compose -p autotrader-dev ...` in `/path/to/Autotrader/dev`
- Docker socket access may require approval in this environment.

## Usual Entry Points
- CLI: `app/main.py`
- Trading loop: `app/agents/trader.py`
- Backtesting: `app/backtest/agent_engine.py`
- Config: `config/config.yaml`

## Backtest Cases (Dev)
- Case 1: `rl_policy` only
- Case 2: `rl_policy_fees` only
- Case 3: both RL policies enabled
- Keep tracked configs unchanged; use temporary config copies for each run when possible.

## Dev AI Filter (Dynamic Symbols)
- Replaced heuristic scanner with AI filter (`app/data/ai_filter.py`) that scores the full Alpaca US universe.
- AI filter trains on Alpaca historical bars (IEX feed) and stores model at `/data/ai_symbol_filter.pt`.
- `config/config.yaml` uses `data.dynamic_symbols.ai_filter.*` and `data.symbols: []` (no default symbols).

## Dev Backtest Data
- Backtests use `/data` CSVs and `backtest.symbols_source: data_dir`.
- Full-universe Alpaca ingest for 2 months is running via `python -m app.main ingest`.

## Recent Tasks
- Created dev worktree on `master` under `/path/to/Autotrader/dev`.
- Live worktree remains on `v1.0`.
- Enabled a strategy-level pending-order guard to skip signal evaluation while orders are open.
- Cleared stale open-order metric labels so Grafana reflects only current pending orders.
- Added broker-aware shorting guard and dev trading limits.
- Added AI filter for dynamic symbol selection in dev.
- Policy: update `PROJECT_CONTEXT.md` and push to all branches after actions >10 seconds.
- Rebuilt and redeployed all live services via Docker Compose.
- Rebuilt and redeployed live v1 services after dashboard updates.
- Rebuilt and redeployed live v1 services to apply Grafana table fixes.
- Rebuilt and redeployed live v1 services to apply Open Orders axis scaling.
- Rebuilt and redeployed live v1 services to apply autoscale changes on time series panels.
- Enabled Grafana provisioning reloads and restarted Grafana for live and dev stacks.
- Rebuilt and redeployed live v1 services after enabling the pending-order strategy guard.
- Rebuilt and redeployed live v1 services after clearing stale open-order metrics.
- Verified Grafana datasource queries show no open orders and position metrics align with Alpaca.
- Created branch `v2.0` from `master`, deployed it live, and synced dev-trained models into `/path/to/Autotrader/models`.
- Stopped the dev backtest container so only the live v2.0 agent runs.
- Stopped the old GPU learner container and restarted the v2.0 stack so only v2.0 services remain.
- Added a safe fallback when the AI filter module is missing to keep live services from crashing.
- Rebuilt and restarted v2.0 services after guarding the AI filter import.
- Added the AI filter module to v2.0 so the live dynamic symbol filter runs.
- Rebuilt and restarted v2.0 services after adding the AI filter module.
- Ported dev ingestion change to load Alpaca universe and added backtest case config files.
- Ported dev backtest and ingest output artifacts into v2.0.
- Rebuilt and restarted v2.0 services after porting dev artifacts.
- Added log output when AI filter runs to confirm live usage.
- Rebuilt and restarted v2.0 services after adding AI filter logging.
- Added AI filter heartbeat logging every 30s when recent data is available.
- Rebuilt and restarted v2.0 services after adding AI filter heartbeat logs.
- Added a pre-run AI filter log to confirm execution start.
- Rebuilt and restarted v2.0 services after adding AI filter pre-run logs.
- Raised the AI filter universe cap and set refresh interval to 1 minute.
- Rebuilt and restarted v2.0 services after AI filter cadence updates.
- Enabled online updates for the AI symbol filter.
- Rebuilt and restarted v2.0 services after enabling AI filter online updates.
- Increased AI filter online update steps and max symbols for continuous training, then redeployed v2.0.
- Added logging for news catalyst cache refreshes and redeployed v2.0.
- Synced news cache refresh to 1 minute to align with AI filter cadence.
- Rebuilt and restarted v2.0 services after syncing news refresh cadence.
- Added news-aware features to the AI symbol filter.
- Rebuilt and restarted v2.0 services after adding news-aware AI filter features.
- Updated architecture diagram to reflect AI filter using news.
- Fixed indentation regression in trader loop that caused restarts.
- Rebuilt and restarted v2.0 services after fixing trader loop indentation.
- Added logging for news catalyst cache refreshes.
- Updated architecture diagram to reflect AI filter, ingestion, and online updates.
- Increased AI filter online update steps and max symbols for continuous training.

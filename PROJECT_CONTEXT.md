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

## Recent Tasks
- Created dev worktree on `master` under `/path/to/Autotrader/dev`.
- Live worktree remains on `v1.0`.
- Enabled a strategy-level pending-order guard to skip signal evaluation while orders are open.
- Policy: update `PROJECT_CONTEXT.md` and push to all branches after actions >10 seconds.
- Rebuilt and redeployed all live services via Docker Compose.
- Rebuilt and redeployed live v1 services after dashboard updates.
- Rebuilt and redeployed live v1 services to apply Grafana table fixes.
- Rebuilt and redeployed live v1 services to apply Open Orders axis scaling.
- Rebuilt and redeployed live v1 services to apply autoscale changes on time series panels.
- Enabled Grafana provisioning reloads and restarted Grafana for live and dev stacks.

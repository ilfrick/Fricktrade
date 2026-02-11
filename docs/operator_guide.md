<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Operator Guide

## Start and Stop
- Full stack (auto GPU/CPU): `./scripts/compose_up.sh --build`
- Full stack (CPU only): `docker compose up -d --build`
- Stop stack: `docker compose down`
- Minimal services (closed markets): autoheal, healthwatch, prometheus, daily-report, tests-when-closed.

## GPU Management
- GPU auto-disables on CUDA errors (latched in `/data/gpu_state.json`)
- Re-enable GPU: `sudo ./scripts/enable_gpu.sh` then restart stack
- Check GPU status: `cat data/gpu_state.json`
- Verify GPU in use: `docker logs fricktrade-trader-1 2>&1 | grep "Using.*device"`

## Health Checks
- API health: `http://localhost:18081/health`
- Grafana: `http://localhost:3002`
- Prometheus: `http://localhost:9090`

## Market Gating and Ops State
- Market hours config: `market.*` in `config/config.yaml`.
- Healthwatch scheduler: `healthwatch.market_shutdown.*`.
- Ops state file (when enabled): `/data/system_state.json`.

## Trading Safety Controls
- Risk limits: `risk.*` and `trading_limits.*`.
- `risk.enabled: false` bypasses risk checks, but broker account flags still block orders.
- Order queue guardrails: `execution.open_orders.*`.
- Strategy performance kill switch: `strategy.performance.*`.
- Manual kill switches: `kill_switch.*`.

## Data and Symbols
- Live data provider: `data.provider` (yfinance/alpaca/brokers). yfinance uses the market-cache service.
- Dynamic symbols: `data.dynamic_symbols.*`.
- AI symbol filter: `data.dynamic_symbols.ai_filter.*` (PPO-based).
- `data.process_on_new_bar_only` skips per-symbol processing when bars have not advanced.
- Market cache staleness: `market_cache.max_age_multiplier` + `market_cache.ignore_staleness`.
- Filtered symbol cache uses per-symbol Redis/file entries when enabled.

## Models and Artifacts
- Trading PPO policy: `/app/models/ppo_policy.zip` (or `learning.registry.active_path`).
- Model registry: `/app/models/model_registry.json` and `/app/models/registry/*`.
- AI symbol filter PPO: `/data/ai_symbol_filter.zip` with `.meta.json`.
- Orchestrator model: `/data/orchestrator_model.pt` (unused when `orchestrator.rl.enabled: false`).

## Advanced Components
- **RegimeHMM** (`learning.regime.enabled`): Detects market volatility regime (low/med/high). Adds `regime`, `regime_name` to market_state.
- **PortfolioOptimizer** (`portfolio.enabled`): Adjusts position sizes based on allocation constraints.
- **EnsemblePredictor** (`orchestrator.ensemble.enabled`): LightGBM-based signal aggregation.
- **TCN Extractor** (`learning.tcn.enabled`): Temporal convolutional network for RL feature extraction.

## Logs and Audit
- Trader logs: `docker compose logs -f trader`.
- Audit/compliance logs: `monitoring.audit.*` and `monitoring.compliance.*`.

## Tuning Trading Frequency (Position Close Rate)

Multiple parameters across different layers control how often positions are closed.
After changing reward parameters the RL model must be retrained for the new behavior
to take effect. Online training will also adapt over subsequent sessions.

### Reward System (`learning.reward`)

| Parameter | Effect |
|-----------|--------|
| `time_penalty_weight` | Penalty per step while in a position. Higher = stronger pressure to exit. Penalty formula: `-time_penalty_weight * minutes_held / time_normalizer`. |
| `time_normalizer` | Number of minutes at which the time penalty reaches `-time_penalty_weight`. Lower = penalty ramps up faster. 390 = one full trading day. |
| `profit_bonus_weight` | Bonus on profitable trade close. Higher = more incentive to take profits quickly rather than hold for bigger gains. Formula: `profit_bonus_weight * realized_pnl / nav_normalizer`. |
| `bar_interval_minutes` | Minutes per bar step. Must match actual data interval. Affects how fast time penalty accumulates per step. |

### Orchestrator (`orchestrator`)

| Parameter | Effect |
|-----------|--------|
| `mode` | `weight` (default) uses `strategy_weights` to blend signals. `direct` picks single winner. |
| `strategy_weights` | Per-strategy weight map. Higher weight = more influence on combined action. |
| `rl.enabled` | Enable RL orchestrator (default: false). When disabled, uses static weight mode. |
| `rl.time_penalty_per_bar` | Per-bar penalty in RL reward (only when RL enabled). |
| `rl.global_time_penalty.*` | Account-level idle penalty (only when RL enabled). |

### Take-Profit (`risk`)

| Parameter | Effect |
|-----------|--------|
| `take_profit_pct` | Full exit at this % gain from entry. Applies to all strategies. |
| `partial_take_profit_pct` | Sell a fraction at this % gain. Fires once per position. |
| `partial_take_profit_ratio` | Fraction to sell at partial TP (0.5 = half). |

### Strategy Parameters (`strategy.params`)

| Parameter | Effect |
|-----------|--------|
| `exit_threshold_pct` | Intraday momentum: exit when price change falls below this. Higher = more aggressive exits. |
| `position_horizon_minutes` | Max intended hold time for intraday momentum. Lower = tighter hold window. |
| `trend_following.exit_pct` | Trend following exit threshold. Higher = exits on smaller reversals. |

### Risk Manager (`risk`)

| Parameter | Effect |
|-----------|--------|
| `hard_stop_pct` | Force exit when position drops this much. Tighter = faster exit on losers. |
| `trailing_stop_pct` | Trailing stop from peak price. Tighter = locks in gains faster, exits on smaller pullbacks. |
| `circuit_breaker_drawdown_pct` | Stops all trading at this account drawdown level. |

### Guardrail (`learning.guardrail`)

| Parameter | Effect |
|-----------|--------|
| `exit_threshold_pct` | Lower = guardrail more willing to confirm sell/exit signals from the RL agent. |
| `mode` | `confirm` requires both RL agent and guardrail to agree. `override` lets guardrail force exits independently. |

### Recommended Approach

For the biggest impact on closing frequency:

1. Raise `time_penalty_weight` (e.g. 0.5 -> 1.0-1.5).
2. Lower `time_normalizer` (e.g. 390 -> 195).
3. Tighten `trailing_stop_pct` (e.g. 0.7 -> 0.3-0.5) for immediate mechanical exits.
4. Retrain the RL model after config changes.
5. Let online training adapt over subsequent sessions.

### Current Global Defaults (Optimized for Frequent Trades)

The global defaults in `config/config.yaml` are tuned for frequent profitable
trades (suited for accounts allowed to day trade, e.g. Higher).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning.reward.time_penalty_weight` | 1.5 | 30-min hold costs -0.375 reward, strongly discouraging multi-hour holds. |
| `learning.reward.time_normalizer` | 120 | Full penalty at 2 hours instead of a full day. |
| `learning.reward.profit_bonus_weight` | 3.0 | Bigger carrot for taking profits. |
| `risk.trailing_stop_pct` | 0.35 | Locks in gains faster, cuts losers at half the pullback. |
| `risk.hard_stop_pct` | 0.8 | Exit losers at -0.8%. |
| `orchestrator.rl.time_penalty_per_bar` | 0.15 | Orchestrator penalizes hold-heavy strategies 3x more. |
| `orchestrator.rl.global_time_penalty.enabled` | true | Penalizes account-wide inactivity. |
| `orchestrator.rl.global_time_penalty.max_idle_minutes` | 15 | Penalty starts after 15 min idle. |
| `orchestrator.rl.global_time_penalty.penalty_scale` | 0.03 | Triple the idle penalty magnitude. |
| `learning.guardrail.params.exit_threshold_pct` | 0.2 | Match strategy exit thresholds so guardrail doesn't block exits. |
| `strategy.params.exit_threshold_pct` | 0.15 | Exit on smaller reversals. |
| `strategy.params.trend_following.exit_pct` | 0.15 | Exit on smaller counter-trend moves. |

### Per-Account Overrides

Accounts with different risk profiles or regulatory constraints (e.g. PDT) can
override `risk` and `reward` parameters. Overrides are defined in
`brokers.alpaca.accounts[]` in `config/config.yaml` and are deep-merged onto the
global defaults at startup.

Each account entry can contain:
- `risk: { ... }` — overrides keys in the global `risk` section for that account's
  `RiskManager`.
- `reward: { ... }` — overrides keys in `learning.reward` for that account's
  orchestrator `RewardConfig`.

The account `name` must match the env-var account name (from `ALPACA_ACCOUNT_NAMES`)
so the YAML overrides are merged onto the env-var credentials automatically.

**Current per-account overrides:**

| Account | Parameter | Value | Rationale |
|---------|-----------|-------|-----------|
| Realistic ($250, PDT-constrained) | `risk.trailing_stop_pct` | 0.5 | Wider stops to avoid burning PDT allowances on marginal trades. |
| Realistic | `risk.hard_stop_pct` | 1.0 | More room before force-exiting losers. |
| Realistic | `reward.time_penalty_weight` | 0.8 | Less pressure to close positions quickly. |
| Realistic | `reward.time_normalizer` | 240 | Full penalty at 4 hours, not 2. |
| Higher ($2585, day trading allowed) | *(uses global defaults)* | — | Aggressive turnover is appropriate. |

**PDT note:** Accounts under $25k equity are limited to 3 day trades per 5 rolling
business days. The Realistic account overrides reduce trade frequency to conserve
PDT allowances.

**Tuning:** If turnover is too aggressive (many small losses from premature exits),
dial `time_penalty_weight` back to 1.0 or raise `time_normalizer` to 180. These
can be set globally or per-account.

## Daily Reporting Email
- Daily report email uses Alertmanager SMTP settings from `.env` (do not commit secrets).
- Required keys are listed in `.env.example`.

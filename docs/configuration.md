<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Configuration

## Purpose
Centralize runtime settings for all subsystems.

## Location
- Main file: `config/config.yaml`.
- Supports `${ENV_VAR}` interpolation.

## Sources and Precedence
- Defaults live in code; `config/config.yaml` is the primary user-edited source of truth.
- `.env` only affects values referenced via `${ENV_VAR}` (plus services that explicitly read `.env`, e.g. Alertmanager SMTP rendering).
- API `/config/update` writes YAML updates into `config/config.yaml`; there is no implicit merge with `.env`.

## Key Sections
- `app.*` (logging, intervals)
- `market.*` (venues, hours, symbol venue mapping)
  - `market.default_symbol_venue` (fallback venue for unmapped symbols)
  - `market.symbol_venues` (manual symbol -> venue mapping)
  - `market.symbol_venues_auto` (auto-refresh mapping from broker)
  - `market.symbol_sectors` (symbol -> sector mapping for exposure caps)
  - `market.extended_hours.enabled` (trade during configured extended hours)
  - `market.venues[].trading_hours.extended_open/extended_close` (extended session window)
- `data.*` (symbols, dynamic scan, sources)
- `data.quality.*` (OHLCV validation + data quality reports)
- `data.adjustments.*` (optional split/dividend adjustment files)
- `news.*`
- `strategy.*`
- `orchestrator.*` (including `orchestrator.mode` and `orchestrator.rl.*`)
- Orchestrator features include `cash_pct` and `buying_power_pct` (both scaled by equity).
- `risk.*`
- `trading_limits.*`
- `execution.*`
- `execution.brokers.*` (multi-broker routing)
- `learning.*`
- `brokers.*`
- `brokers.<name>.accounts[]` (optional multi-account entries; each becomes a broker instance)
- `backtest.*`
- `monitoring.*`
- `checkpointing.*` (periodic state checkpoints + retention)

## Editing Configuration
- Edit `config/config.yaml` directly or use the web UI at `/ui`.
- Most keys support `${ENV_VAR}` interpolation.
- The web UI rejects unknown keys and shows an error instead of applying changes.
- After changes, rebuild/restart containers to apply: `docker compose up -d --build`.

## Common Toggles
- Strategy selection: `strategy.name` or `strategy.names`.
- Dynamic symbols: `data.dynamic_symbols.enabled`.
- Live market data provider: `data.provider` (yfinance, alpaca, or brokers). yfinance uses the market-cache service.
- Market cache: `market_cache.*` (Redis + file fallback, cache-only reads, batch size).
- AI filter: `data.dynamic_symbols.ai_filter.enabled` (PPO-based by default).
- AI filter model: `data.dynamic_symbols.ai_filter.model_type` and `data.dynamic_symbols.ai_filter.rl.*`.
- AI filter provider: `data.dynamic_symbols.ai_filter.provider` (alpaca or yfinance).
- AI filter coverage filter: `data.dynamic_symbols.ai_filter.coverage_filter` (drop symbols without bars).
- AI filter cached symbols: `data.dynamic_symbols.ai_filter.use_cached_symbols`.
- Keras return overlay: `data.dynamic_symbols.ai_filter.keras_returns.*` (optional scoring feature). Requires `TF_USE_LEGACY_KERAS=1` environment variable for compatibility.
- Data quality validation: `data.quality.enabled`.
- Data adjustments: `data.adjustments.enabled`.
- Broker universe: `data.dynamic_symbols.universe: brokers_active` (Alpaca active universe today).
- Universe price filter: `data.dynamic_symbols.universe_price_filter` (limits broker universe by price_min and cash cap).
- Cash-aware cap: `data.dynamic_symbols.cash_aware` + `data.dynamic_symbols.cash_cap_mode`
  (`cash` limits by available cash; `risk` also caps to max position size).
- News LLM catalyst gate: `news.llm.enabled` (Ollama service on `http://ollama:11434` by default).
- Multi-broker news: `news.provider: brokers`.
- Strategy performance reporting + kill switch: `strategy.performance.*`.
- Online learning: `learning.online.enabled`.
- Model registry: `learning.registry.enabled`.
- Active model pointer: `learning.registry.active_path` + `learning.registry.use_active`.
- Model publish mode: `learning.registry.publish_mode` (best or latest).
- Drift monitoring: `learning.drift.enabled`.
- Online learning ops-state gating: `learning.online.respect_ops_state`.
- Decision audit logs: `monitoring.audit.enabled`.
- Compliance exports: `monitoring.compliance.enabled`.
- Audit/compliance retention: `monitoring.audit.retention_days` + `monitoring.compliance.retention_days`.
- Audit/compliance signing: `monitoring.audit.signing.*` + `monitoring.compliance.signing.*`.
- Checkpointing: `checkpointing.enabled`.
- Multi-broker routing: `execution.brokers.enabled`.
- Auto-split routing across brokers: `execution.brokers.routing.mode: auto_split`.
- Multi-account env auto-detect: `ALPACA_API_KEYS`/`ALPACA_API_SECRETS` or `ALPACA_API_KEY_1` +
  `ALPACA_API_SECRET_1`, and `IBKR_CLIENT_IDS` or `IBKR_CLIENT_ID_1` (optional `IBKR_ACCOUNT_ID_*`).
- Execution algos: `execution.algos.enabled`.
- Adaptive execution: `execution.algos.adaptive.enabled`.
- Impact model: `execution.impact.*`.
- Retry policy: `execution.retry.*`.
- Volatility targeting: `risk.vol_targeting.enabled`.
- VaR/CVaR gating: `risk.var.enabled`.
- Exposure caps: `risk.exposure_caps.enabled`.
- Kill switch profiles: `risk.kill_switch_profiles.enabled`.
- Stress haircuts: `risk.stress.enabled`.
- Liquidity haircuts: `risk.liquidity_haircut.enabled`.
- Minimum order price: `trading_limits.min_price`.
- Signal bias guard: `strategy.signal_bias_guard.*` (blocks counter-trend actions).
- RL signal features: `learning.features.include_signal_features` and `learning.features.signal_interval`.
- Benchmarks: `benchmarking.*` (walk-forward + stress scenarios, bootstrap CI, MC stress, output/plot/PDF paths).
- Market-based stack sleep/wake: `healthwatch.market_shutdown.*` (stops services when markets are closed).
- `healthwatch.market_shutdown.keep_services` should include `tests-when-closed` so closed-market tests can run.
- Ops state file: `healthwatch.market_shutdown.write_state` and `healthwatch.market_shutdown.state_path`.
- Manual kill switches: `kill_switch.*` (force sleep or force liquidation with interlock).
- Daily top movers report: `reports.daily_top_movers.*` (daily winners, email, training data export; includes `feed`).
- Daily top movers signals/news: `reports.daily_top_movers.signal_thresholds.*` and `reports.daily_top_movers.news.*`.
- Daily top movers email overrides: `reports.daily_top_movers.email.smtp_require_tls`.
- Email body is saved under `/data/reports/daily_top_movers/<YYYY-MM-DD>/_email/` as `.txt` and `.html`.
- Alertmanager SMTP/recipient settings are read from `.env` and rendered into a local config file (do not commit secrets): `ALERTMANAGER_SMTP_SMARTHOST`, `ALERTMANAGER_SMTP_FROM`, `ALERTMANAGER_SMTP_USERNAME`, `ALERTMANAGER_SMTP_PASSWORD`, `ALERTMANAGER_SMTP_REQUIRE_TLS`, `ALERTMANAGER_SMTP_HELLO`, `ALERTMANAGER_EMAIL_TO`.
- When no trades or skip metrics are recorded, the report infers a reason via Prometheus gauges
  (active symbol, open orders, positions).
- Metrics now include representative absolute moves (open->close, runup, drawdown) along with %.
- Decision traces are read from `reports.daily_top_movers.decision_trace.*` to explain skip causes.

# Configuration

## Purpose
Centralize runtime settings for all subsystems.

## Location
- Main file: `config/config.yaml`.
- Supports `${ENV_VAR}` interpolation.

## Key Sections
- `app.*` (logging, intervals)
- `market.*` (venues, hours, symbol venue mapping)
  - `market.default_symbol_venue` (fallback venue for unmapped symbols)
  - `market.symbol_venues` (manual symbol -> venue mapping)
  - `market.symbol_venues_auto` (auto-refresh mapping from broker)
- `data.*` (symbols, dynamic scan, sources)
- `news.*`
- `strategy.*`
- `orchestrator.*` (including `orchestrator.mode` and `orchestrator.rl.*`)
- `risk.*`
- `trading_limits.*`
- `execution.*`
- `execution.brokers.*` (multi-broker routing)
- `learning.*`
- `brokers.*`
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
- Live market data provider: `data.provider` (yfinance, alpaca, or brokers).
- AI filter: `data.dynamic_symbols.ai_filter.enabled`.
- Multi-broker universe: `data.dynamic_symbols.universe: brokers_active`.
- Universe price filter: `data.dynamic_symbols.universe_price_filter` (limits broker universe by price_min and cash cap).
- Cash-aware cap: `data.dynamic_symbols.cash_aware` + `data.dynamic_symbols.cash_cap_mode`
  (`cash` limits by available cash; `risk` also caps to max position size).
- News LLM catalyst gate: `news.llm.enabled` (Ollama service on `http://ollama:11434` by default).
- Multi-broker news: `news.provider: brokers`.
- Strategy performance reporting + kill switch: `strategy.performance.*`.
- Online learning: `learning.online.enabled`.
- Checkpointing: `checkpointing.enabled`.
- Multi-broker routing: `execution.brokers.enabled`.
- Execution algos: `execution.algos.enabled`.
- Volatility targeting: `risk.vol_targeting.enabled`.
- Market-based stack sleep/wake: `healthwatch.market_shutdown.*` (stops services when markets are closed).
- Manual kill switches: `kill_switch.*` (force sleep or force liquidation with interlock).
- Daily top movers report: `reports.daily_top_movers.*` (daily winners, email, training data export; includes `feed`).
- Daily top movers signals/news: `reports.daily_top_movers.signal_thresholds.*` and `reports.daily_top_movers.news.*`.
- Daily top movers email overrides: `reports.daily_top_movers.email.smtp_require_tls`.
- Email body is saved under `/data/reports/daily_top_movers/<YYYY-MM-DD>/_email/`.
- When no trades or skip metrics are recorded, the report infers a reason via Prometheus gauges
  (active symbol, open orders, positions).
- Metrics now include representative absolute moves (open->close, runup, drawdown) along with %.
- Decision traces are read from `reports.daily_top_movers.decision_trace.*` to explain skip causes.

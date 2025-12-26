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
- `orchestrator.*` (including `orchestrator.rl.*`)
- `risk.*`
- `trading_limits.*`
- `execution.*`
- `learning.*`
- `brokers.*`
- `backtest.*`
- `monitoring.*`
- `checkpointing.*` (periodic state checkpoints + retention)

## Editing Configuration
- Edit `config/config.yaml` directly or use the web UI at `/ui`.
- Most keys support `${ENV_VAR}` interpolation.
- After changes, rebuild/restart containers to apply: `docker compose up -d --build`.

## Common Toggles
- Strategy selection: `strategy.name` or `strategy.names`.
- Dynamic symbols: `data.dynamic_symbols.enabled`.
- AI filter: `data.dynamic_symbols.ai_filter.enabled`.
- Online learning: `learning.online.enabled`.
- Checkpointing: `checkpointing.enabled`.

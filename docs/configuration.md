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

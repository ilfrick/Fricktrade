<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick -->

# Data Pipeline

## Purpose
Acquire bars, scan symbols, and support AI filtering.

## Implementation
- Download (yfinance): `app/data/downloader.py`.
- Alpaca ingestion: `app/data/ingestion.py`.
- Scanner (snapshots): `app/data/scanner.py`.
- News catalysts: `app/data/news.py`.
- AI symbol filter (PPO): `app/data/ai_filter.py`.
- Market cache: `app/data/market_cache.py` + `app/data/market_cache_service.py`.

## Symbol Filtering
The scanner (`app/data/scanner.py`) automatically excludes problematic symbol types:
- **Preferred shares**: `.PR`, `.PRA` through `.PRZ`
- **Units/Rights/Warrants**: `.U`, `.UN`, `.RT`, `.W`, `.WS`, `.WT`
- **Class shares**: `.A`, `.B`, `.C`
- **SPAC warrants**: 5+ character symbols ending in W or U (e.g., NHICW, RFAIU)
- **ADRs**: Symbols ending in Y (except whitelisted: SONY, BKSY, RELY, TORY, LAZY)

Legitimate tickers that would otherwise be excluded are in `_WHITELIST`:
`SNOW`, `FLOW`, `GROW`, `MENU`, `GURU`, `CREW`, `META`, `MSCI`, `ROKU`, `DOCU`, `DKNG`.

## Configuration
`config/config.yaml`:
- `data.provider` (yfinance/alpaca/brokers; yfinance uses batched caching).
- `data.process_on_new_bar_only` (skip per-symbol processing when bar timestamps have not advanced).
- `market_cache.*` (Redis + file cache for live yfinance bars; optional cache-only reads)
  - `market_cache.cache_only: true` disables direct yfinance calls in the trader/AI filter
  - `market_cache.max_age_multiplier` scales staleness thresholds for cached bars
  - `market_cache.ignore_staleness: false` enforces staleness checks instead of reusing old bars
  - `market_cache.filtered_symbols.enabled` controls caching of AI-filter symbol lists
  - Filtered symbol cache is per symbol (Redis + file) and used by the trader when enabled
- `data.interval`
- `data.lookback_days`
- `data.symbols`
- `data.output_dir`
- `data.sources.*`
- `data.quality.*` (OHLCV validation + per-symbol quality report)
- `data.adjustments.*` (optional split/dividend adjustment files)
- `data.dynamic_symbols.*`
- `data.dynamic_symbols.universe` (`brokers_active` uses Alpaca active universe today and merges positions/orders)
- `data.dynamic_symbols.ai_filter.provider` (alpaca or yfinance bars for AI scoring)
- `data.dynamic_symbols.ai_filter.coverage_filter` (drop symbols without bars)
- `data.dynamic_symbols.ai_filter.use_cached_symbols` (prefer cached filtered symbols when available)
- `data.dynamic_symbols.universe_price_filter` (filters broker universe by price_min and cash cap)
- `data.dynamic_symbols.cash_aware` and `data.dynamic_symbols.cash_cap_mode`
  (caps candidates to <= buying power; when buying power <= 0 or cap < price_min, only positions/open orders remain)
- `data.dynamic_symbols.max_symbols` is capped to the tradeable universe size, but can grow to include positions/open orders.
- `news.*`
  - `news.provider: brokers` aggregates catalysts across enabled brokers (Alpaca-backed today)
  - `news.llm.*` enables optional Ollama catalyst gating per headline
- Held positions and open-order symbols are always included in dynamic symbol results, even if scanner filters would exclude them.
- When `data.adjustments.enabled` is true, ingestion looks for per-symbol adjustment files in
  `data.adjustments.dir` named `<SYMBOL>.csv` with columns `Datetime`, `split_ratio`, and `dividend`.

## Ingest Example
```bash
docker compose run --rm trader python3 -m app.main ingest --config /app/config/config.yaml
```

## Download Example
```bash
docker compose run --rm trader python3 -m app.main download --config /app/config/config.yaml --symbols AAPL MSFT
```

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
- AI symbol filter: `app/data/ai_filter.py`.

## Configuration
`config/config.yaml`:
- `data.interval`
- `data.lookback_days`
- `data.symbols`
- `data.output_dir`
- `data.sources.*`
- `data.quality.*` (OHLCV validation + per-symbol quality report)
- `data.adjustments.*` (optional split/dividend adjustment files)
- `data.dynamic_symbols.*`
- `data.dynamic_symbols.universe` (`brokers_active` merges enabled broker universes with positions/orders)
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

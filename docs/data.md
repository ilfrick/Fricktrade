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
- `data.dynamic_symbols.*`
- `data.dynamic_symbols.universe_price_filter` (filters broker universe by price_min and cash cap)
- `data.dynamic_symbols.cash_aware` and `data.dynamic_symbols.cash_cap_mode`
  (caps candidates to <= buying power; when buying power <= 0 or cap < price_min, only positions/open orders remain)
- `data.dynamic_symbols.max_symbols` is capped to the tradeable universe size, but can grow to include positions/open orders.
- `news.*`
- Held positions and open-order symbols are always included in dynamic symbol results, even if scanner filters would exclude them.

## Ingest Example
```bash
docker compose run --rm trader python3 -m app.main ingest --config /app/config/config.yaml
```

## Download Example
```bash
docker compose run --rm trader python3 -m app.main download --config /app/config/config.yaml --symbols AAPL MSFT
```

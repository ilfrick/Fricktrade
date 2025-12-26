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
- `data.dynamic_symbols.universe` (`brokers_active` merges enabled broker universes with positions/orders)
- `data.dynamic_symbols.universe_price_filter` (filters broker universe by price_min and cash cap)
- `news.*`
  - `news.provider: brokers` aggregates catalysts across enabled brokers (Alpaca-backed today)

## Ingest Example
```bash
docker compose run --rm trader python3 -m app.main ingest --config /app/config/config.yaml
```

## Download Example
```bash
docker compose run --rm trader python3 -m app.main download --config /app/config/config.yaml --symbols AAPL MSFT
```

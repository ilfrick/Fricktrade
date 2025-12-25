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
- `news.*`

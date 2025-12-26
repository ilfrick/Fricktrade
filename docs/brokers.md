# Brokers

## Purpose
Provide a unified API for Alpaca and IBKR.

## Implementations
- Alpaca: `app/brokers/alpaca.py`.
- IBKR: `app/brokers/ibkr.py`.
- Interface: `app/brokers/base.py`.

## Selection
- Controlled by `brokers.ibkr.enabled`.
- Alpaca uses `TRADING_MODE=paper` to decide paper vs live.

## Credentials
- Alpaca: `ALPACA_API_KEY`, `ALPACA_API_SECRET`
- IBKR: configured under `brokers.ibkr.*`

## Configuration
`config/config.yaml`:
- `brokers.alpaca.*`
- `brokers.ibkr.*`
- `brokers.<name>.fees.*`

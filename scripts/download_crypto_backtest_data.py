#!/usr/bin/env python3
"""Download crypto 5m bars from Binance for backtesting.

Usage (inside container):
    docker exec fricktrade-trader-1 python3 /app/scripts/download_crypto_backtest_data.py \
        --config /app/config/config.yaml --days 90

Usage (host, with venv):
    python3 scripts/download_crypto_backtest_data.py --config config/config.yaml --days 90
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Top crypto symbols by market cap / liquidity
DEFAULT_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
    "DOGE/USD", "AVAX/USD", "LINK/USD", "DOT/USD", "UNI/USD",
    "LTC/USD", "BCH/USD", "AAVE/USD", "FIL/USD", "RENDER/USD",
    "ONDO/USD", "NEAR/USD", "ATOM/USD", "ARB/USD", "OP/USD",
]


def _to_binance_symbol(sym: str) -> str:
    base, quote = sym.split("/")
    return f"{base}USDT"


def _fetch_klines(client, binance_sym: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch all klines in the range, paginating as needed."""
    all_klines = []
    current_start = start_ms
    while current_start < end_ms:
        klines = client.get_klines(
            symbol=binance_sym,
            interval=interval,
            startTime=current_start,
            endTime=end_ms,
            limit=1000,
        )
        if not klines:
            break
        all_klines.extend(klines)
        # Next page starts after the last kline's open time
        last_open = klines[-1][0]
        if last_open <= current_start:
            break
        current_start = last_open + 1
    return all_klines


def download_symbol(client, symbol: str, days: int, out_dir: Path, interval: str = "5m") -> Path | None:
    binance_sym = _to_binance_symbol(symbol)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    try:
        klines = _fetch_klines(client, binance_sym, interval, start_ms, end_ms)
    except Exception as exc:
        log.warning("Failed to fetch %s (%s): %s", symbol, binance_sym, exc)
        return None

    if not klines:
        log.warning("No data for %s", symbol)
        return None

    rows = []
    for k in klines:
        rows.append({
            "Datetime": pd.Timestamp(k[0], unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="Datetime").sort_values("Datetime")

    filename = f"{symbol.replace('/', '_')}_{interval}.csv"
    path = out_dir / filename
    df.to_csv(path, index=False)
    log.info("  %s: %d bars (%.1f days) → %s", symbol, len(df), days, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--symbols", nargs="*", help="Override symbol list")
    parser.add_argument("--out-dir", help="Output directory (default: backtest.data_dir from config)")
    args = parser.parse_args()

    from app.utils.config import load_config
    from app.data.binance_market_data import _build_binance_client_from_cfg

    cfg = load_config(args.config)
    client = _build_binance_client_from_cfg(cfg)
    if client is None:
        log.error("Could not build Binance client — check config")
        sys.exit(1)

    symbols = args.symbols or DEFAULT_SYMBOLS
    out_dir = Path(args.out_dir or cfg.get("backtest", {}).get("data_dir", "/data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %d crypto symbols, %d days of 5m bars → %s", len(symbols), args.days, out_dir)
    downloaded = 0
    for sym in symbols:
        path = download_symbol(client, sym, args.days, out_dir)
        if path:
            downloaded += 1
        time.sleep(0.5)  # Rate limiting

    log.info("Done: %d/%d symbols downloaded", downloaded, len(symbols))


if __name__ == "__main__":
    main()

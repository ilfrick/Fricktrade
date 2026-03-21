#!/usr/bin/env python3
"""Download US equity 5m bars from yfinance for backtesting.

yfinance limits 5-minute intraday data to ~60 calendar days, so this script
downloads the maximum available window (60 days) rather than the 90 days
available for daily bars.

Usage (inside container):
    docker exec fricktrade-trader-1 python3 /app/scripts/download_equity_backtest_data.py

Usage (host, with venv):
    python3 scripts/download_equity_backtest_data.py --days 60 --out-dir /data
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# 20 major US equities by market cap / liquidity
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "META", "AMZN",
    "GOOGL", "TSLA", "JPM", "V", "UNH",
    "HD", "NFLX", "CRM", "COST", "AMD",
    "INTC", "BAC", "XOM", "JNJ", "PG",
]

# yfinance hard limit for 5m intraday data
MAX_5M_DAYS = 60


def download_symbol(
    symbol: str,
    days: int,
    out_dir: Path,
    interval: str = "5m",
) -> Path | None:
    """Download 5m bars for a single equity symbol and save to CSV.

    yfinance caps intraday 5m history at ~60 days. Requests exceeding this
    are silently clamped by Yahoo, so we enforce the limit client-side.
    """
    effective_days = min(days, MAX_5M_DAYS)

    try:
        ticker = yf.Ticker(symbol)
        # Use period= instead of start/end — yfinance is stricter with
        # explicit date ranges and often rejects valid 5m windows.
        df = ticker.history(
            interval=interval,
            period=f"{effective_days}d",
            auto_adjust=True,
            actions=False,
        )
    except Exception as exc:
        log.warning("Failed to fetch %s: %s", symbol, exc)
        return None

    if df is None or df.empty:
        log.warning("No data returned for %s", symbol)
        return None

    # Normalise columns to Open, High, Low, Close, Volume
    col_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    keep_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep_cols]

    # Forward-fill then back-fill OHLC; fill missing volume with 0
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0.0)

    # Drop duplicate timestamps, sort
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Convert index to timezone-naive UTC string for CSV portability
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "Datetime"

    filename = f"{symbol}_5m.csv"
    path = out_dir / filename
    df.to_csv(path)
    trading_days = df.index.normalize().nunique()
    log.info(
        "  %s: %d bars (%d trading days) -> %s",
        symbol, len(df), trading_days, path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 5m equity bars from yfinance for backtesting.",
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help=f"Days of history (capped at {MAX_5M_DAYS} for 5m data). Default: 60",
    )
    parser.add_argument(
        "--symbols", nargs="*",
        help="Override default symbol list (space-separated)",
    )
    parser.add_argument(
        "--out-dir", default="/data",
        help="Output directory for CSVs (default: /data)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between symbols to avoid rate-limiting (default: 1.0)",
    )
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_days = min(args.days, MAX_5M_DAYS)
    if args.days > MAX_5M_DAYS:
        log.warning(
            "Requested %d days but yfinance caps 5m data at %d days; using %d",
            args.days, MAX_5M_DAYS, MAX_5M_DAYS,
        )

    log.info(
        "Downloading %d equity symbols, %d days of 5m bars -> %s",
        len(symbols), effective_days, out_dir,
    )

    downloaded = 0
    failed: list[str] = []
    for i, sym in enumerate(symbols, 1):
        log.info("[%d/%d] Fetching %s ...", i, len(symbols), sym)
        path = download_symbol(sym, effective_days, out_dir)
        if path:
            downloaded += 1
        else:
            failed.append(sym)
        # Rate-limit between requests
        if i < len(symbols):
            time.sleep(args.delay)

    log.info("Done: %d/%d symbols downloaded successfully", downloaded, len(symbols))
    if failed:
        log.warning("Failed symbols: %s", ", ".join(failed))


if __name__ == "__main__":
    main()

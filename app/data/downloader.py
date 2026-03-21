# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from pathlib import Path
import logging
import time
from datetime import datetime

import pandas as pd

from app.data.yfinance_utils import fetch_yfinance_bars


def _download_with_retries(
    symbol: str,
    lookback_days: int,
    interval: str,
    proxy: str | None,
    start: str | None = None,
    end: str | None = None,
    retries: int = 3,
):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            bars, _ = fetch_yfinance_bars(
                [symbol],
                lookback_days,
                interval,
                batch_size=1,
                lowercase=False,
                drop_zero_volume=False,
                proxy=proxy,
                start=start,
                end=end,
                use_ticker_history=True,
            )
            data = bars.get(symbol)
            if data is not None and not data.empty:
                return data
        except Exception as exc:
            last_err = exc
            logging.warning("yfinance download failed for %s (attempt %d/%d): %s", symbol, attempt, retries, exc)
        time.sleep(attempt * 2)
    if last_err:
        raise last_err
    return None


def _clamp_lookback(interval: str, lookback_days: int) -> int:
    # yfinance limits for intraday intervals
    if interval == "1m":
        return min(lookback_days, 7)
    if interval in {"2m", "5m", "15m", "30m"}:
        return min(lookback_days, 60)
    if interval in {"60m", "90m", "1h"}:
        return min(lookback_days, 730)
    return lookback_days


def download_yfinance(
    symbols: list[str],
    interval: str,
    lookback_days: int,
    out_dir: str,
    proxy: str = "",
    rate_limit_seconds: int = 2,
    start: str = "",
    end: str = "",
) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    files = []
    clamped_days = _clamp_lookback(interval, lookback_days)
    start = start or None
    end = end or None
    for symbol in symbols:
        data = _download_with_retries(
            symbol,
            clamped_days,
            interval,
            proxy,
            start=start,
            end=end,
        )
        if data is None or data.empty:
            continue
        ordered = ["Open", "High", "Low", "Close", "Volume"]
        if all(col in data.columns for col in ordered):
            data = data[ordered]
        file_path = out_path / f"{symbol.replace('/', '_').replace('.', '_')}_{interval}.csv"
        data.to_csv(file_path, index_label="Datetime")
        files.append(file_path)
        time.sleep(rate_limit_seconds)
    return files


def _alpaca_imports():
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as exc:
        raise ImportError("alpaca-py is required for Alpaca downloads") from exc
    return StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit


def _alpaca_timeframe(interval: str, TimeFrame, TimeFrameUnit):
    if interval.endswith("m"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)
    if interval.endswith("h"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
    if interval.endswith("d"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Day)
    return TimeFrame(1, TimeFrameUnit.Day)


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def download_alpaca_bars(
    symbols: list[str],
    interval: str,
    out_dir: str,
    api_key: str,
    api_secret: str,
    start: str = "",
    end: str = "",
    rate_limit_seconds: int = 1,
) -> list[Path]:
    if not api_key or not api_secret:
        logging.warning("Alpaca API credentials missing; skipping download")
        return []
    StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit = _alpaca_imports()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    client = StockHistoricalDataClient(api_key, api_secret)
    timeframe = _alpaca_timeframe(interval, TimeFrame, TimeFrameUnit)
    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)
    files = []
    for symbol in symbols:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
            adjustment="raw",
        )
        try:
            data = client.get_stock_bars(req).df
        except Exception as exc:
            logging.warning("Alpaca bars download failed for %s: %s", symbol, exc)
            time.sleep(rate_limit_seconds)
            continue
        if data is None or data.empty:
            time.sleep(rate_limit_seconds)
            continue
        if isinstance(data.index, pd.MultiIndex):
            data = data.copy()
            data.index = data.index.get_level_values(-1)
        data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        ordered = ["Open", "High", "Low", "Close", "Volume"]
        if all(col in data.columns for col in ordered):
            data = data[ordered]
        file_path = out_path / f"{symbol.replace('/', '_').replace('.', '_')}_{interval}.csv"
        data.to_csv(file_path, index_label="Datetime")
        files.append(file_path)
        time.sleep(rate_limit_seconds)
    return files

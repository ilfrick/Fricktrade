# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf


def fetch_yfinance_bars(
    symbols: list[str],
    lookback_days: int,
    interval: str,
    *,
    batch_size: int = 100,
    lowercase: bool = False,
    drop_zero_volume: bool = False,
    delay_seconds: float = 0.0,
) -> tuple[dict[str, pd.DataFrame], list[str]]: # Changed return type hint
    symbols = [s for s in symbols if s]
    if not symbols:
        return {}, [] # Return empty list of failed symbols

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    failed_symbols: list[str] = [] # Initialize list to track failed symbols

    for idx in range(0, len(symbols), batch_size):
        chunk = symbols[idx : idx + batch_size]
        try:
            data = yf.download(
                tickers=" ".join(chunk),
                period=f"{lookback_days}d",
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logging.warning("yfinance download failed for %d symbols: %s", len(chunk), exc)
            failed_symbols.extend(chunk) # Add all symbols in chunk to failed_symbols
            continue
        
        if data is None or data.empty:
            logging.warning("yfinance returned no data for %d symbols in chunk.", len(chunk))
            failed_symbols.extend(chunk) # Add all symbols in chunk to failed_symbols
            continue

        if getattr(data.columns, "nlevels", 1) > 1:
            for symbol in chunk:
                if symbol not in data.columns.get_level_values(1):
                    logging.warning("yfinance: no price data found for symbol '%s' in chunk.", symbol)
                    failed_symbols.append(symbol) # Add individual failed symbol
                    continue
                frame = data.xs(symbol, level=1, axis=1)
                cleaned = _clean_yfinance_frame(frame, lowercase=lowercase, drop_zero_volume=drop_zero_volume)
                if cleaned is not None:
                    bars_by_symbol[symbol] = cleaned
                else:
                    logging.warning("yfinance: cleaned data is None for symbol '%s'.", symbol)
                    failed_symbols.append(symbol) # Add if cleaned data is None
        elif len(chunk) == 1:
            symbol = chunk[0]
            cleaned = _clean_yfinance_frame(data, lowercase=lowercase, drop_zero_volume=drop_zero_volume)
            if cleaned is not None:
                bars_by_symbol[symbol] = cleaned
            else:
                logging.warning("yfinance: cleaned data is None for single symbol '%s'.", symbol)
                failed_symbols.append(symbol) # Add if cleaned data is None
        
        if delay_seconds > 0 and idx + batch_size < len(symbols):
            time.sleep(delay_seconds)
            
    return bars_by_symbol, failed_symbols # Return both


def _clean_yfinance_frame(
    frame: pd.DataFrame,
    *,
    lowercase: bool,
    drop_zero_volume: bool,
) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    if lowercase:
        frame = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        ohlc_cols = ("open", "high", "low", "close")
        volume_col = "volume"
    else:
        if "close" in frame.columns:
            frame = frame.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
        ohlc_cols = ("Open", "High", "Low", "Close")
        volume_col = "Volume"
    for col in ohlc_cols:
        if col in frame.columns:
            frame[col] = frame[col].ffill().bfill()
    if volume_col in frame.columns:
        frame[volume_col] = frame[volume_col].fillna(0.0)
    if any(col in frame.columns and frame[col].isna().any() for col in ohlc_cols):
        return None
    if drop_zero_volume and volume_col in frame.columns and (frame[volume_col] <= 0).all():
        return None
    return frame.sort_index()

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
        
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            nlevels = getattr(data.columns, "nlevels", 1)
            logging.debug(
                "yfinance chunk=%s rows=%d cols=%d nlevels=%d",
                chunk,
                len(data),
                data.shape[1],
                nlevels,
            )
            logging.debug("yfinance index sample for chunk %s: %s", chunk, data.index[:3])
            if len(chunk) <= 5:
                logging.debug("yfinance columns sample for chunk %s: %s", chunk, list(data.columns)[:10])

        try:
            if getattr(data.columns, "nlevels", 1) > 1:
                # Handle MultiIndex DataFrame (multiple symbols).
                nlevels = int(data.columns.nlevels)
                level_sets = [set(data.columns.get_level_values(i)) for i in range(nlevels)]
                for symbol in chunk:
                    level_idx = next((i for i, vals in enumerate(level_sets) if symbol in vals), None)
                    if level_idx is None:
                        logging.warning("yfinance: no price data found for symbol '%s' in chunk.", symbol)
                        failed_symbols.append(symbol)
                        continue
                    frame = data.xs(symbol, level=level_idx, axis=1)
                    cleaned = _clean_yfinance_frame(frame, lowercase=lowercase, drop_zero_volume=drop_zero_volume)
                    if cleaned is not None:
                        bars_by_symbol[symbol] = cleaned
                    else:
                        logging.warning("yfinance: cleaned data is None for symbol '%s'.", symbol)
                        failed_symbols.append(symbol)
            elif len(chunk) == 1:
                # Handle single-symbol DataFrame
                symbol = chunk[0]
                cleaned = _clean_yfinance_frame(data, lowercase=lowercase, drop_zero_volume=drop_zero_volume)
                if cleaned is not None:
                    bars_by_symbol[symbol] = cleaned
                else:
                    logging.warning("yfinance: cleaned data is None for single symbol '%s'.", symbol)
                    failed_symbols.append(symbol)
            else:
                # This case implies data was not empty, but neither MultiIndex nor single-symbol.
                # This could happen if yf.download returns a malformed DataFrame for multiple symbols.
                logging.warning(
                    "yfinance returned unexpected data structure for %d symbols in chunk. Adding to failed_symbols.",
                    len(chunk)
                )
                failed_symbols.extend(chunk)

        except (AttributeError, KeyError, TypeError) as e:
            logging.error(
                "yfinance data processing error for chunk (symbols: %s): %s. Adding all to failed_symbols.",
                ", ".join(chunk),
                e
            )
            failed_symbols.extend(chunk)
        
        if delay_seconds > 0 and idx + batch_size < len(symbols):
            time.sleep(delay_seconds)
            
    logging.debug("Final failed_symbols after processing all chunks: %s", failed_symbols) # Add this
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

#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Hourly collector for crypto return-ranker training data.

For each symbol in data.crypto_symbols:
  1. Fetches 5m bars from Alpaca (24h lookback, 24/7).
  2. Labels each bar with the 60-minute forward log-return.
  3. Appends (features, label) rows to data/training/crypto_ranker_{YYYY-MM-DD}.csv.
  4. Optionally triggers retraining if a sufficient number of rows have been collected.

No market-hours gate — crypto trades 24/7.

Cron (hourly):
  0 * * * *  cd /home/nicola/Fricktrade && python3 scripts/collect_crypto_training_data.py

Usage:
  python3 scripts/collect_crypto_training_data.py [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collect_crypto_training")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _PROJECT_ROOT / "data" / "training"
FORWARD_BARS = 12       # 12 × 5m = 60-minute forward return
MIN_BARS_BEFORE = 20    # minimum history for feature extraction
SAMPLE_EVERY_BARS = 6   # sample every 30 minutes (6 × 5m)

FEATURE_NAMES = [
    "ret_5m", "ret_15m", "ret_60m",
    "vol_5m", "vol_15m", "vol_60m",
    "rsi_14",
    "vwap_dist_pct",
    "rel_volume",
    "log_dollar_volume",
    "high_low_range_pct",
    "atr_pct",
]


def _fetch_alpaca_bars(symbol: str, api_key: str, api_secret: str,
                        lookback_hours: int = 24) -> list[dict] | None:
    """Fetch 5m bars from Alpaca REST API. Returns list of OHLCV dicts."""
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient  # type: ignore[import]
        from alpaca.data.requests import CryptoBarsRequest  # type: ignore[import]
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore[import]

        client = CryptoHistoricalDataClient(api_key, api_secret)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=lookback_hours)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_dt,
            end=end_dt,
        )
        bars_df = client.get_crypto_bars(req).df
        if bars_df is None or bars_df.empty:
            return None
        # Reset multi-index if present
        if bars_df.index.names and bars_df.index.names[0] == "symbol":
            bars_df = bars_df.reset_index(level="symbol", drop=True)
        bars_df = bars_df.sort_index()
        records = []
        for ts, row in bars_df.iterrows():
            records.append({
                "ts": ts,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            })
        return records
    except ImportError:
        log.warning("alpaca-py not installed; using yfinance fallback for %s", symbol)
        return _fetch_yfinance_bars(symbol, lookback_hours)
    except Exception as exc:
        log.warning("Alpaca bars fetch failed for %s: %s", exc, symbol)
        return _fetch_yfinance_bars(symbol, lookback_hours)


def _fetch_yfinance_bars(symbol: str, lookback_hours: int = 24) -> list[dict] | None:
    """Yfinance fallback for crypto bars. Uses Yahoo symbol conversion."""
    try:
        import yfinance as yf  # type: ignore[import]
        yf_sym = symbol.replace("/", "-")
        # yfinance period max for 5m is 60d
        period = "1d" if lookback_hours <= 24 else "2d"
        df = yf.download(yf_sym, period=period, interval="5m", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        records = []
        for ts, row in df.iterrows():
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            records.append({
                "ts": ts,
                "open": float(row.get("Open", 0)),
                "high": float(row.get("High", 0)),
                "low": float(row.get("Low", 0)),
                "close": float(row.get("Close", 0)),
                "volume": float(row.get("Volume", 0)),
            })
        return records
    except Exception as exc:
        log.warning("yfinance fallback failed for %s: %s", symbol, exc)
        return None


def _extract_features(bars: list[dict], idx: int) -> dict | None:
    """Extract feature vector at bar index idx."""
    if idx < MIN_BARS_BEFORE:
        return None
    closes = [b["close"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    volumes = [b["volume"] for b in bars]

    c = closes[idx]
    if c <= 0:
        return None

    # Returns
    def safe_ret(i, j):
        if i < 0 or j < 0 or closes[i] <= 0:
            return 0.0
        return math.log(closes[j] / closes[i]) if closes[i] > 0 else 0.0

    ret_5m = safe_ret(idx - 1, idx)
    ret_15m = safe_ret(max(idx - 3, 0), idx)
    ret_60m = safe_ret(max(idx - 12, 0), idx)

    # Volatility (std of 5m returns)
    def vol_window(w):
        rets = [math.log(closes[i] / closes[i-1]) for i in range(max(idx-w+1, 1), idx+1)
                if closes[i-1] > 0]
        if len(rets) < 2:
            return 0.0
        m = sum(rets) / len(rets)
        return math.sqrt(sum((r - m)**2 for r in rets) / len(rets))

    vol_5m = vol_window(5)
    vol_15m = vol_window(15)
    vol_60m = vol_window(60)

    # RSI(14)
    period = min(14, idx)
    gains = []
    losses = []
    for i in range(max(idx - period, 0), idx):
        diff = closes[i+1] - closes[i]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss > 0 else 50.0

    # VWAP distance
    vwap_window = min(idx, 78)  # ~6.5h in 5m bars
    tv = sum(volumes[max(idx-vwap_window, 0):idx+1])
    if tv > 0:
        vwap = sum(
            ((highs[i] + lows[i] + closes[i]) / 3.0) * volumes[i]
            for i in range(max(idx-vwap_window, 0), idx+1)
        ) / tv
        vwap_dist_pct = (c - vwap) / vwap * 100.0 if vwap > 0 else 0.0
    else:
        vwap_dist_pct = 0.0

    # Relative volume
    vol_avg = sum(volumes[max(idx-12, 0):idx+1]) / max(idx - max(idx-12, 0) + 1, 1)
    rel_volume = volumes[idx] / vol_avg if vol_avg > 0 else 1.0

    # Log dollar volume
    log_dv = math.log(c * volumes[idx] + 1.0)

    # High-low range
    hl_range_pct = (highs[idx] - lows[idx]) / c * 100.0 if c > 0 else 0.0

    # ATR(14)
    trs = []
    for i in range(max(idx-14, 1), idx+1):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1]),
        )
        trs.append(tr)
    atr = sum(trs) / len(trs) if trs else 0.0
    atr_pct = atr / c * 100.0 if c > 0 else 0.0

    return {
        "ret_5m": round(ret_5m, 6),
        "ret_15m": round(ret_15m, 6),
        "ret_60m": round(ret_60m, 6),
        "vol_5m": round(vol_5m, 6),
        "vol_15m": round(vol_15m, 6),
        "vol_60m": round(vol_60m, 6),
        "rsi_14": round(rsi, 3),
        "vwap_dist_pct": round(vwap_dist_pct, 4),
        "rel_volume": round(rel_volume, 4),
        "log_dollar_volume": round(log_dv, 4),
        "high_low_range_pct": round(hl_range_pct, 4),
        "atr_pct": round(atr_pct, 4),
    }


def _maybe_retrain(output_dir: str, data_dir: str) -> None:
    """Trigger retraining if the combined training data has grown significantly."""
    files = sorted(Path(output_dir).glob("crypto_ranker_*.csv"))
    # Remove empty files
    for f in files:
        if f.stat().st_size < 500:
            f.unlink()
            log.info("Removed empty training file: %s", f.name)
    try:
        from app.signals import return_ranker_train as rrt  # type: ignore[import]
        model_path = str(Path(data_dir) / "crypto_ranker.pkl")
        rrt.train_return_ranker(
            str(output_dir),
            model_path,
            max_age_days=3,
            file_prefix="crypto_ranker_",
        )
        log.info("Crypto ranker retrained → %s", model_path)
    except Exception as exc:
        log.warning("Crypto ranker retraining skipped: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--lookback-hours", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    crypto_symbols: list[str] = list(data_cfg.get("crypto_symbols", []))
    if not crypto_symbols:
        log.info("No crypto_symbols configured. Exiting.")
        return

    alpaca_cfg = cfg.get("brokers", {}).get("alpaca", {})
    api_key = alpaca_cfg.get("api_key", "") or ""
    api_secret = alpaca_cfg.get("api_secret", "") or ""
    # Resolve ${ENV_VAR} syntax
    import os, re
    for attr in ("api_key", "api_secret"):
        val = alpaca_cfg.get(attr, "") or ""
        m = re.match(r"^\$\{(.+)\}$", val)
        if m:
            api_key_val = os.environ.get(m.group(1), "")
            if attr == "api_key":
                api_key = api_key_val
            else:
                api_secret = api_key_val

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = output_dir / f"crypto_ranker_{today}.csv"

    fieldnames = ["symbol", "ts"] + FEATURE_NAMES + ["forward_ret_60m"]
    write_header = not out_path.exists()

    rows_written = 0
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for symbol in crypto_symbols:
            bars = _fetch_alpaca_bars(symbol, api_key, api_secret, args.lookback_hours)
            if not bars or len(bars) < MIN_BARS_BEFORE + FORWARD_BARS:
                log.debug("Insufficient bars for %s (%s)", symbol, len(bars) if bars else 0)
                continue

            # Sample every SAMPLE_EVERY_BARS bars (avoid redundant rows)
            for idx in range(MIN_BARS_BEFORE, len(bars) - FORWARD_BARS, SAMPLE_EVERY_BARS):
                feats = _extract_features(bars, idx)
                if feats is None:
                    continue
                c_now = bars[idx]["close"]
                c_fwd = bars[idx + FORWARD_BARS]["close"]
                if c_now <= 0:
                    continue
                fwd_ret = math.log(c_fwd / c_now) if c_fwd > 0 else 0.0
                row: dict = {
                    "symbol": symbol,
                    "ts": bars[idx]["ts"].isoformat() if hasattr(bars[idx]["ts"], "isoformat")
                          else str(bars[idx]["ts"]),
                    **feats,
                    "forward_ret_60m": round(fwd_ret, 6),
                }
                writer.writerow(row)
                rows_written += 1

    log.info("crypto_training: %d rows written → %s", rows_written, out_path)

    if rows_written >= 100:
        _maybe_retrain(str(output_dir), str(_PROJECT_ROOT / "data"))


if __name__ == "__main__":
    main()

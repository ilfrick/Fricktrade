#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Post-session data collector for return ranker model training.

For each symbol active during today's (or the specified) session:
  1. Reads the decision trace to find the set of symbols and their timestamps.
  2. Fetches 5m OHLCV bars from yfinance — the same source the live AI filter uses.
  3. For every (symbol, decision_ts) pair inside market hours, extracts the exact
     same feature vector that return_ranker.py will use at inference time.
  4. Labels each row with the actual log-return 60 minutes later (forward_ret_60m).
  5. Appends all (features, label) rows to data/training/return_ranker_{date}.csv.

Run daily 5–10 minutes after US market close:
  cron: 10 22 * * 1-5    (22:10 CET = 16:10 ET)

Usage:
  python3 scripts/collect_return_ranker_data.py [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

# Make app/ importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.signals.return_ranker import FEATURE_NAMES, extract_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("collect_return_ranker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ET = ZoneInfo("US/Eastern")
MARKET_OPEN_H, MARKET_OPEN_M = 9, 30
MARKET_CLOSE_H, MARKET_CLOSE_M = 16, 0
FORWARD_BARS = 12          # 12 × 5m = 60-minute forward return
MIN_BARS_BEFORE_TS = 20    # minimum history needed to compute features
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = _PROJECT_ROOT / "data" / "reports" / "decision_trace"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "training"
# Decimate trace: sample at most one record per symbol per SAMPLE_EVERY_SECONDS
SAMPLE_EVERY_SECONDS = 300   # 5-minute grid matches bar frequency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Session date YYYY-MM-DD (ET). Default: today.")
    parser.add_argument("--trace-dir", default=str(TRACE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--lookback-days", type=int, default=5,
                        help="Days of 5m bars to fetch from yfinance (covers intraday window)")
    parser.add_argument("--news-lookback-hours", type=int, default=12,
                        help="Hours of news history to fetch for catalyst/recency features")
    args = parser.parse_args()

    tz_date = datetime.now(ET).strftime("%Y-%m-%d") if not args.date else args.date
    log.info("collect_return_ranker: session_date=%s", tz_date)

    # Trace file is named by UTC date; market open (9:30 ET) = 14:30 UTC = same UTC date
    target_dt = datetime.strptime(tz_date, "%Y-%m-%d").replace(tzinfo=ET)
    utc_date = target_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    trace_path = Path(args.trace_dir) / f"{utc_date}.jsonl"

    if not trace_path.exists():
        log.error("Trace file not found: %s", trace_path)
        sys.exit(1)

    market_open = target_dt.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    market_close = target_dt.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)

    # Step 1: read trace → collect (symbol, ts) pairs inside market hours
    log.info("Reading trace: %s", trace_path)
    symbol_timestamps = _read_trace(trace_path, market_open, market_close)
    if not symbol_timestamps:
        log.error("No decision records found in market-hours window")
        sys.exit(1)
    symbols = sorted(symbol_timestamps)
    log.info("Found %d active symbols, %d total (symbol, ts) pairs",
             len(symbols), sum(len(v) for v in symbol_timestamps.values()))

    # Step 2: fetch 5m bars for all symbols (same yfinance call as ai_filter)
    log.info("Fetching 5m bars from yfinance for %d symbols ...", len(symbols))
    bars_by_symbol = _fetch_bars(symbols, lookback_days=args.lookback_days)
    log.info("Got bars for %d / %d symbols", len(bars_by_symbol), len(symbols))

    # Step 2b: fetch news features (catalyst, article_count, recency_hours)
    # Uses the same Alpaca news API as the live system; falls back to zeros on error.
    news_features_map = _fetch_news(symbols, args.news_lookback_hours, as_of=market_close)

    # Merge trace-extracted catalyst flags (ground truth from live LLM decisions)
    trace_news = _extract_news_from_trace(trace_path, market_open, market_close)
    for sym, tn in trace_news.items():
        if sym not in news_features_map:
            news_features_map[sym] = tn
        elif tn.get("catalyst") and not news_features_map[sym].get("catalyst"):
            news_features_map[sym]["catalyst"] = True

    log.info("News features fetched for %d symbols (catalyst=%d)",
             len(news_features_map),
             sum(1 for v in news_features_map.values() if v.get("catalyst")))

    # Step 3 + 4: extract features and compute forward-return labels
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"return_ranker_{tz_date}.csv"

    rows_written = 0
    rows_skipped = 0

    fieldnames = ["symbol", "ts"] + FEATURE_NAMES + ["forward_ret_60m"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for symbol in symbols:
            frame = bars_by_symbol.get(symbol)
            if frame is None or frame.empty:
                rows_skipped += len(symbol_timestamps[symbol])
                continue

            frame = _normalize_frame(frame)
            timestamps = sorted(symbol_timestamps[symbol])
            nf = news_features_map.get(symbol)

            for ts in timestamps:
                row = _make_row(symbol, ts, frame, news_features=nf)
                if row is None:
                    rows_skipped += 1
                    continue
                writer.writerow(row)
                rows_written += 1

    log.info("Done: %d rows written, %d skipped → %s", rows_written, rows_skipped, out_path)
    if rows_written >= 200:
        _maybe_retrain(args.output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_trace(
    trace_path: Path,
    market_open: datetime,
    market_close: datetime,
) -> dict[str, list[datetime]]:
    """Return {symbol: [ts, ...]} for records inside market hours, sampled at 5m grid."""
    result: dict[str, list[datetime]] = {}
    last_sample: dict[str, datetime] = {}

    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_str = rec.get("ts")
            symbol = rec.get("symbol", "")
            if not ts_str or not symbol:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_et = ts.astimezone(ET)
            if ts_et < market_open or ts_et >= market_close:
                continue
            # 5-minute grid decimation
            prev = last_sample.get(symbol)
            if prev and (ts - prev).total_seconds() < SAMPLE_EVERY_SECONDS:
                continue
            last_sample[symbol] = ts
            result.setdefault(symbol, []).append(ts)

    return result


def _extract_news_from_trace(
    trace_path: Path,
    market_open: datetime,
    market_close: datetime,
) -> dict[str, dict]:
    """Extract {symbol: {catalyst: bool, ...}} from trace signal_inputs."""
    result: dict[str, dict] = {}
    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            symbol = rec.get("symbol")
            si = rec.get("signal_inputs", {})
            catalyst = si.get("catalyst", False)
            if symbol and catalyst and symbol not in result:
                result[symbol] = {
                    "catalyst": True,
                    "article_count": 0,
                    "recency_hours": 12.0,
                }
    return result


def _fetch_news(symbols: list[str], lookback_hours: int = 12, as_of: datetime | None = None) -> dict[str, dict]:
    """Fetch per-symbol news features from Alpaca using the live system's config."""
    try:
        import os, re, yaml
        cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
        # Load .env so ${VAR} references in config.yaml resolve
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        raw = cfg_path.read_text()
        raw = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), raw)
        cfg = yaml.safe_load(raw)
        news_cfg = (cfg.get("data", {}).get("dynamic_symbols", {})
                      .get("ai_filter", {}).get("news", {}))
        if not news_cfg.get("enabled", False):
            log.info("News disabled in config — using zero news features")
            default = {"catalyst": False, "article_count": 0, "recency_hours": float(lookback_hours)}
            return {s: dict(default) for s in symbols}
        from app.brokers.config_utils import get_alpaca_account_cfg
        alpaca_cfg = get_alpaca_account_cfg(cfg.get("brokers", {}))
        api_key = alpaca_cfg.get("api_key", "")
        api_secret = alpaca_cfg.get("api_secret", "")
        from app.data.news import fetch_news_features
        # Build llm_cfg; when running on host (not Docker), replace
        # Docker-internal hostname (ollama) with localhost.
        llm_raw = dict(news_cfg.get("llm", {}) or {})
        if llm_raw.get("enabled"):
            llm_url = str(llm_raw.get("base_url", "http://localhost:11434"))
            llm_url = llm_url.replace("://ollama:", "://localhost:")
            llm_raw["base_url"] = llm_url
        return fetch_news_features(
            symbols,
            provider=str(news_cfg.get("provider", "alpaca")),
            base_url=str(news_cfg.get("base_url", "https://data.alpaca.markets")),
            api_key=api_key,
            api_secret=api_secret,
            lookback_hours=lookback_hours,
            keywords=list(news_cfg.get("keywords", [])),
            llm_cfg=llm_raw,
            timeout_seconds=int(news_cfg.get("timeout_seconds", 10)),
            retries=int(news_cfg.get("retries", 2)),
            as_of=as_of,
        )
    except Exception as exc:
        log.warning("News feature fetch failed: %s — using zeros", exc)
        default = {"catalyst": False, "article_count": 0, "recency_hours": float(lookback_hours)}
        return {s: dict(default) for s in symbols}


def _fetch_bars(symbols: list[str], lookback_days: int = 5) -> dict:
    """Fetch 5m OHLCV bars from yfinance — same call as ai_filter._fetch_bars_yfinance."""
    try:
        from app.data.yfinance_utils import fetch_yfinance_bars
        bars, _ = fetch_yfinance_bars(
            symbols,
            lookback_days,
            "5m",
            batch_size=100,
            lowercase=True,
            drop_zero_volume=False,
        )
        return bars
    except Exception as exc:
        log.error("yfinance fetch failed: %s", exc)
        return {}


def _normalize_frame(frame) -> "pd.DataFrame":
    """Ensure DatetimeIndex with UTC timezone for bar lookup."""
    import pandas as pd
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame = frame.copy()
            frame.index = pd.to_datetime(frame.index)
        except Exception:
            return frame
    if frame.index.tzinfo is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    return frame.sort_index()


def _make_row(symbol: str, ts: datetime, frame, news_features: dict | None = None) -> dict | None:
    """Extract feature vector and forward-return label for one (symbol, ts) pair."""
    import pandas as pd

    close_col = frame["close"].astype(float) if "close" in frame.columns else None
    if close_col is None:
        return None

    ts_utc = ts.astimezone(timezone.utc).replace(tzinfo=timezone.utc)

    # Slice bars up to (and including) this timestamp
    past = frame[frame.index <= ts_utc]
    if len(past) < MIN_BARS_BEFORE_TS:
        return None

    # Feature extraction — same function as inference, with live news features
    features = extract_features(past, window=20, interval="5m", news_features=news_features)
    if features is None:
        return None

    # Forward return: bar at ts + FORWARD_BARS (60 min at 5m)
    future_ts = ts_utc + timedelta(minutes=5 * FORWARD_BARS)
    future = frame[frame.index <= future_ts]
    if len(future) == 0 or future.index[-1] < ts_utc + timedelta(minutes=5 * (FORWARD_BARS // 2)):
        return None     # fewer than half the expected forward bars — skip

    close_now = float(past["close"].iloc[-1])
    close_future = float(future["close"].iloc[-1])
    if close_now <= 0:
        return None
    forward_ret_60m = float(np.log(close_future / close_now))

    row: dict = {"symbol": symbol, "ts": ts_utc.isoformat()}
    for name, val in zip(FEATURE_NAMES, features):
        row[name] = round(float(val), 8)
    row["forward_ret_60m"] = round(forward_ret_60m, 8)
    return row


def _maybe_retrain(data_dir: str) -> None:
    """Trigger model retraining now that new data is available."""
    try:
        import yaml
        cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
        model_path = "data/return_ranker.pkl"  # default
        try:
            with cfg_path.open() as f:
                cfg = yaml.safe_load(f)
            rr_cfg = (cfg.get("data", {}).get("dynamic_symbols", {})
                        .get("ai_filter", {}).get("return_ranker", {}))
            model_path = str(rr_cfg.get("model_path", model_path))
        except Exception:
            pass
        # Remove empty training CSVs (header-only, < 500 bytes)
        for f in Path(data_dir).glob("return_ranker_*.csv"):
            if f.stat().st_size < 500:
                f.unlink()
                log.info("Removed empty training file: %s", f.name)

        from app.signals.return_ranker_train import train_return_ranker
        max_age_days = int(rr_cfg.get("max_age_days", 3))
        log.info("Triggering return_ranker retraining → %s (max_age_days=%d)", model_path, max_age_days)
        ok = train_return_ranker(data_dir, model_path, max_age_days=max_age_days)
        log.info("Retraining %s", "succeeded" if ok else "failed (insufficient data?)")
    except Exception as exc:
        log.warning("Retraining skipped: %s", exc)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from app.brokers.config_utils import get_alpaca_account_cfg
from app.data.market_cache import MarketCache, build_market_cache_config, interval_to_seconds
from app.data.scanner import load_universe
from app.data.yfinance_utils import fetch_yfinance_bars
from app.utils.config import load_config
from app.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Market cache service")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        cfg["app"]["log_level"],
        file_path=log_cfg.get("file_path"),
        max_bytes=int(log_cfg.get("max_bytes", 5_000_000)),
        backup_count=int(log_cfg.get("backup_count", 5)),
    )

    cache_cfg = build_market_cache_config(cfg.get("market_cache", {}))
    if not cache_cfg.enabled:
        logging.info("Market cache disabled; exiting.")
        return
    _configure_yfinance_cache(cache_cfg.file_dir)

    cache = MarketCache(cache_cfg.redis_url, cache_cfg.file_dir, ignore_staleness=cache_cfg.ignore_staleness)
    dyn_cfg = cfg.get("data", {}).get("dynamic_symbols", {}) or {}
    universe_cfg = dyn_cfg.get("universe", "alpaca_active")
    max_universe = int(dyn_cfg.get("max_universe", 50000))
    intervals = _intervals_from_cfg(cfg)
    lookbacks = _lookbacks_from_cfg(cfg)
    batch_size = cache_cfg.batch_size
    
    yfinance_source_cfg = {}
    for source in cfg.get("data", {}).get("sources", []):
        if source.get("provider") == "yfinance":
            yfinance_source_cfg = source
            break

    # To implement dynamic filtering without a permanent blacklist:
    # We maintain a temporary set of failed symbols for a certain duration.
    # Symbols are removed from this set after a "retry_after" period.
    temporary_failed_symbols: dict[str, datetime] = {} # {symbol: failed_timestamp}
    # This value will determine how long a failed symbol is temporarily excluded.
    # Using rate_limit_seconds from yfinance config as a proxy for retry_after.
    # If not set, default to 5 minutes (300 seconds).
    retry_after_seconds = yfinance_source_cfg.get("rate_limit_seconds", 300) * 2 # Give it a bit more time

    last_run: dict[str, datetime] = {}
    while True:
        # Clean up temporary_failed_symbols: re-add symbols that are past their retry_after time
        now = datetime.utcnow()
        symbols_to_retry = [
            s for s, timestamp in temporary_failed_symbols.items() 
            if (now - timestamp).total_seconds() > retry_after_seconds
        ]
        for s in symbols_to_retry:
            temporary_failed_symbols.pop(s)
            logging.info("Market cache: Re-attempting previously failed symbol '%s'.", s)
            
        universe = _resolve_universe(cfg, universe_cfg, max_universe)
        if not universe:
            logging.warning("Market cache: universe empty; sleeping.")
            time.sleep(60)
            continue

        # Filter universe based on temporary_failed_symbols
        effective_universe = [s for s in universe if s not in temporary_failed_symbols]
        if len(effective_universe) < len(universe):
            logging.info(
                "Market cache: Temporarily filtered out %d symbols due to previous failures.",
                len(universe) - len(effective_universe)
            )
        
        if not effective_universe:
            logging.warning("Market cache: effective universe empty after filtering; sleeping.")
            time.sleep(60)
            continue
            
        for interval in intervals:
            now = datetime.utcnow()
            interval_seconds = interval_to_seconds(interval)
            last_at = last_run.get(interval)
            if last_at and (now - last_at).total_seconds() < interval_seconds:
                continue
            lookback_days = lookbacks.get(interval, 1)
            # Fetch bars and also get the list of failed symbols from yfinance
            bars, newly_failed_symbols = fetch_yfinance_bars(
                effective_universe,
                lookback_days,
                interval,
                batch_size=batch_size,
                lowercase=False,
                drop_zero_volume=False,
                delay_seconds=5.0,
            )
            # Update temporary_failed_symbols with new failures
            for s in newly_failed_symbols:
                temporary_failed_symbols[s] = now
                logging.warning("Market cache: Temporarily blacklisting symbol '%s' due to yfinance error.", s)

            cache.set_bars(bars, interval, ttl_seconds=interval_seconds)
            last_run[interval] = now
            logging.info(
                "Market cache refreshed interval=%s symbols=%d lookback_days=%d",
                interval,
                len(bars),
                lookback_days,
            )
        sleep_seconds = min(interval_to_seconds(interval) for interval in intervals) if intervals else 60
        time.sleep(max(int(sleep_seconds), 30))


def _intervals_from_cfg(cfg: dict) -> list[str]:
    intervals = set()
    data_cfg = cfg.get("data", {}) or {}
    interval = str(data_cfg.get("interval", "1m"))
    intervals.add(interval)
    ai_cfg = data_cfg.get("dynamic_symbols", {}).get("ai_filter", {}) or {}
    ai_interval = ai_cfg.get("interval")
    if ai_interval:
        intervals.add(str(ai_interval))
    return sorted(intervals)


def _lookbacks_from_cfg(cfg: dict) -> dict[str, int]:
    data_cfg = cfg.get("data", {}) or {}
    interval = str(data_cfg.get("interval", "1m"))
    lookback = int(data_cfg.get("lookback_days", 1))
    ai_cfg = data_cfg.get("dynamic_symbols", {}).get("ai_filter", {}) or {}
    ai_interval = str(ai_cfg.get("interval", interval))
    ai_lookback = int(ai_cfg.get("lookback_days", lookback))
    lookbacks: dict[str, int] = {interval: lookback}
    lookbacks[ai_interval] = max(lookbacks.get(ai_interval, 1), ai_lookback)
    return lookbacks


def _resolve_universe(cfg: dict, universe_cfg: object, max_universe: int) -> list[str]:
    universe_cfg = "alpaca_active" if str(universe_cfg) == "brokers_active" else universe_cfg
    if str(universe_cfg) != "alpaca_active":
        return load_universe("", "", universe_cfg, max_universe=max_universe)
    alpaca_cfg = get_alpaca_account_cfg(cfg)
    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    if not api_key or not api_secret:
        logging.warning("Market cache: Alpaca credentials missing; no universe loaded.")
        return []
    return load_universe(api_key, api_secret, universe_cfg, max_universe=max_universe)


def _configure_yfinance_cache(base_dir: str) -> None:
    cache_dir = Path(base_dir) / "yf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import yfinance as yf

        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir / "tz"))
    except Exception as exc:
        logging.warning("Market cache yfinance cache setup failed: %s", exc)


if __name__ == "__main__":
    main()

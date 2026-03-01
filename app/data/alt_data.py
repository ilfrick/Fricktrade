# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Alternative data providers with in-memory TTL caching.

  1. Fear & Greed Index  — alternative.me (no key)
  2. Crypto open interest via CoinGlass  — requires COINGLASS_API_KEY
  3. SEC EDGAR insider trades  — public API, no key
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, object]] = {}  # key → (expires_ts, value)


def _cached(key: str, ttl: float, factory):
    now = datetime.now(timezone.utc).timestamp()
    if key in _CACHE:
        expires, val = _CACHE[key]
        if now < expires:
            return val
    val = factory()
    _CACHE[key] = (now + ttl, val)
    return val


# ---------------------------------------------------------------------------
# 1. CNN Fear & Greed Index
# ---------------------------------------------------------------------------

def fetch_fear_greed(ttl_seconds: int = 3600) -> Optional[float]:
    """Return Fear & Greed index value (0–100). No API key required.

    Uses alternative.me API: https://api.alternative.me/fng/
    Returns None on failure.
    """
    def _fetch():
        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            observations = data.get("data", [])
            if observations:
                return float(observations[0].get("value", 50))
        except Exception as exc:
            logger.debug("fetch_fear_greed failed: %s", exc)
        return None

    return _cached("fear_greed", float(ttl_seconds), _fetch)


# ---------------------------------------------------------------------------
# 2. CoinGlass open interest
# ---------------------------------------------------------------------------

def fetch_coinglass_oi(
    symbol: str,
    api_key: str,
    base_url: str = "https://open-api.coinglass.com/public/v2",
    ttl_seconds: int = 3600,
) -> Optional[dict]:
    """Return open interest metrics for a crypto symbol from CoinGlass.

    Returns dict with keys: open_interest_usd, change_pct_24h, long_pct, short_pct
    Returns None if the API key is missing or the request fails.
    """
    if not api_key:
        return None

    # Strip exchange suffix: BTC/USD → BTC
    base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
    cache_key = f"coinglass_oi_{base}"

    def _fetch():
        try:
            params = urllib.parse.urlencode({
                "symbol": base,
                "interval": "h1",
                "api_key": api_key,
            })
            url = f"{base_url}/indicator/open_interest?{params}"
            req = urllib.request.Request(url, headers={"coinglassSecret": api_key})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            # CoinGlass wraps results in data[] sorted desc
            entries = data.get("data", [])
            if not entries:
                return None
            latest = entries[0]
            return {
                "open_interest_usd": float(latest.get("openInterest", 0) or 0),
                "change_pct_24h": float(latest.get("openInterestChange24h", 0) or 0),
                "long_pct": float(latest.get("longRatio", 50) or 50),
                "short_pct": float(latest.get("shortRatio", 50) or 50),
            }
        except Exception as exc:
            logger.debug("fetch_coinglass_oi(%s) failed: %s", base, exc)
            return None

    return _cached(cache_key, float(ttl_seconds), _fetch)


# ---------------------------------------------------------------------------
# 3. SEC EDGAR insider trades
# ---------------------------------------------------------------------------

_TICKER_CIK_CACHE: dict[str, Optional[str]] = {}


def _lookup_cik(symbol: str) -> Optional[str]:
    """Resolve ticker to SEC CIK (padded to 10 digits)."""
    if symbol in _TICKER_CIK_CACHE:
        return _TICKER_CIK_CACHE[symbol]
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            tickers = json.loads(resp.read())
        for _, entry in tickers.items():
            if str(entry.get("ticker", "")).upper() == symbol.upper():
                cik = str(entry.get("cik_str", "")).zfill(10)
                _TICKER_CIK_CACHE[symbol] = cik
                return cik
    except Exception as exc:
        logger.debug("CIK lookup for %s failed: %s", symbol, exc)
    _TICKER_CIK_CACHE[symbol] = None
    return None


def fetch_sec_insider_trades(symbol: str, ttl_seconds: int = 86400) -> list[dict]:
    """Return recent Form 4 insider transactions for the given ticker.

    Returns list of dicts: {date, insider, transaction_type, shares, value}
    Returns empty list on failure or if no CIK found.
    """
    cache_key = f"sec_insider_{symbol.upper()}"

    def _fetch():
        cik = _lookup_cik(symbol)
        if not cik:
            return []
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            req = urllib.request.Request(url, headers={"User-Agent": "fricktrade/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                sub_data = json.loads(resp.read())
            filings = sub_data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            reporters = filings.get("reportingOwnerName", filings.get("reporterName", []))
            accessions = filings.get("accessionNumber", [])
            trades = []
            for i, form in enumerate(forms):
                if form != "4":
                    continue
                trades.append({
                    "date": dates[i] if i < len(dates) else "",
                    "insider": reporters[i] if i < len(reporters) else "",
                    "transaction_type": "unknown",  # would need full filing parse
                    "accession": accessions[i] if i < len(accessions) else "",
                    "shares": 0,
                    "value": 0.0,
                })
                if len(trades) >= 20:
                    break
            return trades
        except Exception as exc:
            logger.debug("fetch_sec_insider_trades(%s) failed: %s", symbol, exc)
            return []

    return _cached(cache_key, float(ttl_seconds), _fetch) or []


def insider_sentiment(trades: list[dict]) -> int:
    """Summarise insider trades as +1 (net buying), -1 (net selling), 0 (neutral/unknown)."""
    if not trades:
        return 0
    buys = sum(1 for t in trades if str(t.get("transaction_type", "")).lower() in {"p", "buy", "purchase"})
    sells = sum(1 for t in trades if str(t.get("transaction_type", "")).lower() in {"s", "sell", "sale"})
    if buys > sells:
        return 1
    if sells > buys:
        return -1
    return 0

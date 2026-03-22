# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Binance Futures funding rate poller — exposes leverage sentiment.

Polls the public funding rate endpoint every ~10 minutes and exposes:
  - funding_rate: raw 8h funding rate (typically -0.01% to +0.03%)
  - funding_extreme: float in [-1.0, 1.0]
      +1.0 = extremely positive funding (longs overleveraged → mean reversion buy signal)
      -1.0 = extremely negative funding (shorts overleveraged → caution for buys)
       0.0 = normal funding

Thresholds based on historical distribution:
  |rate| > 0.03% = extreme (top/bottom ~5% of historical)
  |rate| > 0.01% = elevated
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


def _to_futures_symbol(symbol: str) -> str:
    base, _quote = symbol.split("/")
    return f"{base}USDT"


class FundingRatePoller:
    """Polls Binance Futures funding rates and computes extreme-leverage signals."""

    EXTREME_THRESHOLD = 0.0003  # 0.03% per 8h
    ELEVATED_THRESHOLD = 0.0001  # 0.01% per 8h

    def __init__(self, symbols: list[str], poll_interval: int = 600):
        self._symbols = {s: _to_futures_symbol(s) for s in symbols if "/" in s}
        self._reverse_map = {v: k for k, v in self._symbols.items()}
        self._poll_interval = poll_interval
        self._rates: dict[str, float] = {}  # symbol → raw funding rate
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not _REQUESTS_AVAILABLE or not self._symbols:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="funding-rate-poller")
        self._thread.start()
        logger.info("FundingRatePoller started, tracking %d symbols", len(self._symbols))

    def stop(self) -> None:
        self._running = False

    def get_funding_rate(self, symbol: str) -> float:
        """Raw funding rate for symbol. 0.0 if unknown."""
        with self._lock:
            return self._rates.get(symbol, 0.0)

    def get_funding_extreme(self, symbol: str) -> float:
        """Funding extreme score in [-1.0, 1.0].

        Positive = longs overleveraged (pay shorts) → mean reversion opportunity.
        Negative = shorts overleveraged → caution.
        """
        rate = self.get_funding_rate(symbol)
        if abs(rate) < self.ELEVATED_THRESHOLD:
            return 0.0
        # Linear scale from elevated to extreme, capped at 1.0
        sign = 1.0 if rate > 0 else -1.0
        magnitude = abs(rate)
        if magnitude >= self.EXTREME_THRESHOLD:
            return sign * 1.0
        # Scale linearly between elevated and extreme
        scale = (magnitude - self.ELEVATED_THRESHOLD) / (self.EXTREME_THRESHOLD - self.ELEVATED_THRESHOLD)
        return round(sign * scale, 4)

    def get_all_rates(self) -> dict[str, float]:
        with self._lock:
            return dict(self._rates)

    def _run(self) -> None:
        while self._running:
            try:
                self._poll()
            except Exception as exc:
                logger.warning("FundingRatePoller error: %s", exc)
            for _ in range(self._poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _poll(self) -> None:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        for internal_sym, futures_sym in self._symbols.items():
            try:
                resp = requests.get(url, params={"symbol": futures_sym, "limit": 1}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        rate = float(data[-1].get("fundingRate", 0))
                        with self._lock:
                            self._rates[internal_sym] = rate
            except Exception as exc:
                logger.debug("Funding rate fetch error %s: %s", futures_sym, exc)
            time.sleep(0.2)  # rate limit courtesy

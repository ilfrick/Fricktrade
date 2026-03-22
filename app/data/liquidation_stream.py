# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Binance Futures liquidation stream — detects cascade events.

Connects to the all-market force-order stream and tracks cumulative
liquidation volume per symbol in rolling windows.  A cascade is
detected when liquidation volume in a window exceeds a threshold.

Signal:  cascade_score in [-1.0, +1.0]
   +1.0 = heavy long liquidations (longs getting rekt → price falling → MR buy opportunity)
   -1.0 = heavy short liquidations (shorts getting rekt → price rising)
    0.0 = no significant liquidation activity

Falls back gracefully if WebSocket connection fails — score defaults to 0.0.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any

logger = logging.getLogger(__name__)

try:
    import websocket
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    logger.warning("websocket-client not available; liquidation stream disabled")


def _to_futures_symbol(symbol: str) -> str:
    """Convert internal symbol (BTC/USD) to Binance futures format (BTCUSDT)."""
    base, _quote = symbol.split("/")
    return f"{base}USDT"


class LiquidationStream:
    """Streams Binance Futures forced liquidation orders and computes cascade scores.

    Uses the all-market force order stream: wss://fstream.binance.com/ws/!forceOrder@arr
    """

    def __init__(
        self,
        symbols: list[str],
        window_seconds: int = 300,
        cascade_threshold_usd: float = 500_000.0,
    ):
        self._symbols = {s: _to_futures_symbol(s) for s in symbols if "/" in s}
        self._reverse_map = {v: k for k, v in self._symbols.items()}
        self._window_seconds = window_seconds
        self._cascade_threshold = cascade_threshold_usd
        # Per-symbol deques of (timestamp, side, notional_usd)
        self._events: dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self._lock = threading.Lock()
        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not _WS_AVAILABLE or not self._symbols:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="liquidation-stream")
        self._thread.start()
        logger.info("LiquidationStream started, tracking %d symbols", len(self._symbols))

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def get_cascade_score(self, symbol: str) -> float:
        """Get cascade score for a symbol.

        Returns float in [-1.0, 1.0]:
          Positive = long liquidations dominate (price falling, buy opportunity)
          Negative = short liquidations dominate (price rising)
          0.0 = no significant activity
        """
        with self._lock:
            events = self._events.get(symbol)
            if not events:
                return 0.0
            now = time.time()
            cutoff = now - self._window_seconds
            # Sum long and short liquidation volume in window
            long_vol = 0.0
            short_vol = 0.0
            for ts, side, notional in events:
                if ts < cutoff:
                    continue
                if side == "SELL":  # forced SELL = long liquidation
                    long_vol += notional
                else:  # forced BUY = short liquidation
                    short_vol += notional
            total = long_vol + short_vol
            if total < self._cascade_threshold * 0.1:
                return 0.0  # noise
            # Score: positive when longs are being liquidated (price falling)
            imbalance = (long_vol - short_vol) / total if total > 0 else 0.0
            # Scale by how much total volume exceeds threshold
            intensity = min(total / self._cascade_threshold, 3.0) / 3.0
            return round(imbalance * intensity, 4)

    def get_all_scores(self) -> dict[str, float]:
        with self._lock:
            return {sym: self.get_cascade_score(sym) for sym in self._symbols}

    def _run(self) -> None:
        while self._running:
            try:
                self._connect()
            except Exception as exc:
                logger.warning("LiquidationStream connection error: %s", exc)
            if self._running:
                time.sleep(5)

    def _connect(self) -> None:
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            data = json.loads(message)
            order = data.get("o", data)
            futures_sym = order.get("s", "")  # e.g. BTCUSDT
            symbol = self._reverse_map.get(futures_sym)
            if not symbol:
                return
            side = order.get("S", "")  # BUY or SELL
            qty = float(order.get("q", 0))
            price = float(order.get("p", 0))
            notional = qty * price
            with self._lock:
                self._events[symbol].append((time.time(), side, notional))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.debug("Liquidation parse error: %s", exc)

    def _on_error(self, ws: Any, error: Any) -> None:
        logger.debug("LiquidationStream WS error: %s", error)

    def _on_close(self, ws: Any, close_status: Any = None, close_msg: Any = None) -> None:
        logger.debug("LiquidationStream WS closed")

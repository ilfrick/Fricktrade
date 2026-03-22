# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Binance order book depth stream — provides bid/ask imbalance signal.

Connects to Binance WebSocket partial depth streams (@depth10@100ms) for
subscribed symbols and computes a bid/ask imbalance ratio:

    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

Range: -1.0 (all sell pressure) to +1.0 (all buy pressure).
Used by:
  - Peak detection exit: sell when imbalance shifts to sell-heavy (< -0.2)
  - Entry timing: buy when imbalance is bid-heavy (> 0.2) after a dip

Falls back gracefully if WebSocket connection fails — imbalance defaults to 0.0.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Try to import websocket-client (already a dependency via python-binance)
try:
    import websocket
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    logger.warning("websocket-client not available; order book imbalance disabled")


def _to_binance_ws_symbol(symbol: str) -> str:
    """Convert internal symbol (BTC/USD) to Binance WebSocket stream format (btcusdt)."""
    base, _quote = symbol.split("/")
    return f"{base.lower()}usdt"


class OrderBookStream:
    """Streams Binance partial order book depth and computes bid/ask imbalance.

    Uses the combined stream endpoint to subscribe to multiple symbols
    on a single WebSocket connection.
    """

    def __init__(self, symbols: list[str], depth_levels: int = 10, base_url: str = ""):
        self._symbols = [s for s in symbols if "/" in s]  # crypto only
        self._depth_levels = depth_levels
        self._imbalances: dict[str, float] = {}  # symbol → imbalance [-1, 1]
        self._raw_depth: dict[str, dict] = {}  # symbol → {bid_depth, ask_depth}
        self._lock = threading.Lock()
        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._base_url = base_url

    def start(self) -> None:
        """Start the WebSocket depth stream in a background thread."""
        if not _WS_AVAILABLE or not self._symbols:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="order-book-stream")
        self._thread.start()
        logger.info("OrderBookStream started for %d symbols", len(self._symbols))

    def stop(self) -> None:
        """Stop the WebSocket stream."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def get_imbalance(self, symbol: str) -> float:
        """Get the current bid/ask imbalance for a symbol.

        Returns:
            float in [-1.0, 1.0]. 0.0 if no data available.
            Positive = bid-heavy (buy pressure), negative = ask-heavy (sell pressure).
        """
        with self._lock:
            return self._imbalances.get(symbol, 0.0)

    def get_all_imbalances(self) -> dict[str, float]:
        """Get imbalances for all subscribed symbols."""
        with self._lock:
            return dict(self._imbalances)

    def _run(self) -> None:
        """WebSocket event loop with auto-reconnect."""
        while self._running:
            try:
                self._connect()
            except Exception as exc:
                logger.warning("OrderBookStream connection error: %s", exc)
            if self._running:
                time.sleep(5)  # reconnect delay

    def _connect(self) -> None:
        """Connect to Binance combined stream for partial depth."""
        streams = [f"{_to_binance_ws_symbol(s)}@depth{self._depth_levels}@100ms"
                   for s in self._symbols]
        # Binance combined stream endpoint
        if self._base_url and "demo" in self._base_url:
            ws_base = "wss://demo-stream.binance.com"
        else:
            ws_base = "wss://stream.binance.com:9443"
        url = f"{ws_base}/stream?streams={'/'.join(streams)}"

        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_message(self, ws: Any, message: str) -> None:
        """Process incoming depth update."""
        try:
            data = json.loads(message)
            # Combined stream wraps payload in {"stream": "...", "data": {...}}
            payload = data.get("data", data)
            stream = data.get("stream", "")

            # Extract symbol from stream name (e.g., "btcusdt@depth10@100ms")
            ws_sym = stream.split("@")[0] if stream else ""
            if not ws_sym:
                return

            # Find matching internal symbol
            symbol = self._resolve_symbol(ws_sym)
            if not symbol:
                return

            bids = payload.get("bids", [])
            asks = payload.get("asks", [])

            # Sum notional depth (price × qty) for top N levels
            bid_depth = sum(float(b[0]) * float(b[1]) for b in bids[:self._depth_levels])
            ask_depth = sum(float(a[0]) * float(a[1]) for a in asks[:self._depth_levels])

            total = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

            with self._lock:
                self._imbalances[symbol] = imbalance
                self._raw_depth[symbol] = {
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "imbalance": imbalance,
                }

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.debug("OrderBook parse error: %s", exc)

    def _on_error(self, ws: Any, error: Any) -> None:
        logger.debug("OrderBookStream WS error: %s", error)

    def _on_close(self, ws: Any, close_status: Any = None, close_msg: Any = None) -> None:
        logger.debug("OrderBookStream WS closed")

    def _resolve_symbol(self, ws_sym: str) -> str | None:
        """Map Binance WebSocket symbol (btcusdt) back to internal format (BTC/USD)."""
        ws_sym_upper = ws_sym.upper()
        for sym in self._symbols:
            base, _quote = sym.split("/")
            if f"{base}USDT" == ws_sym_upper:
                return sym
        return None

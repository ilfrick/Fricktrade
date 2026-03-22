# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Cross-exchange price divergence — Coinbase vs Binance.

Connects to the Coinbase Exchange WebSocket (free, no auth) and compares
real-time ticker prices against Binance prices from the order book stream.

Signal: divergence_pct per symbol
   Positive = Coinbase price > Binance (Binance lagging, expect upward convergence)
   Negative = Coinbase price < Binance (Binance leading, expect downward convergence)
   0.0 = no divergence or no data

Only covers symbols listed on both exchanges.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import websocket
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False

# Map internal symbols to Coinbase product IDs
_COINBASE_PRODUCTS = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    "DOGE/USD": "DOGE-USD",
    "ADA/USD": "ADA-USD",
    "AVAX/USD": "AVAX-USD",
    "LINK/USD": "LINK-USD",
    "DOT/USD": "DOT-USD",
    "UNI/USD": "UNI-USD",
    "XRP/USD": "XRP-USD",
    "LTC/USD": "LTC-USD",
    "NEAR/USD": "NEAR-USD",
    "FIL/USD": "FIL-USD",
    "AAVE/USD": "AAVE-USD",
    "ATOM/USD": "ATOM-USD",
    "ARB/USD": "ARB-USD",
    "OP/USD": "OP-USD",
    "RENDER/USD": "RNDR-USD",
    "SHIB/USD": "SHIB-USD",
}


class CrossExchangeStream:
    """Streams Coinbase ticker prices and computes divergence vs a reference price source."""

    def __init__(self, symbols: list[str]):
        self._symbols = [s for s in symbols if s in _COINBASE_PRODUCTS]
        self._product_map = {_COINBASE_PRODUCTS[s]: s for s in self._symbols}
        self._coinbase_prices: dict[str, float] = {}
        self._reference_prices: dict[str, float] = {}  # set externally (from Binance)
        self._lock = threading.Lock()
        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not _WS_AVAILABLE or not self._symbols:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="cross-exchange-stream")
        self._thread.start()
        logger.info("CrossExchangeStream started, tracking %d symbols on Coinbase", len(self._symbols))

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def update_reference_price(self, symbol: str, price: float) -> None:
        """Update the Binance reference price for divergence calculation."""
        with self._lock:
            self._reference_prices[symbol] = price

    def get_divergence_pct(self, symbol: str) -> float:
        """Get price divergence percentage (Coinbase - Binance) / Binance * 100.

        Returns 0.0 if no data for either exchange.
        """
        with self._lock:
            cb_price = self._coinbase_prices.get(symbol, 0.0)
            ref_price = self._reference_prices.get(symbol, 0.0)
        if cb_price <= 0 or ref_price <= 0:
            return 0.0
        return round((cb_price - ref_price) / ref_price * 100, 4)

    def get_all_divergences(self) -> dict[str, float]:
        return {sym: self.get_divergence_pct(sym) for sym in self._symbols}

    def _run(self) -> None:
        while self._running:
            try:
                self._connect()
            except Exception as exc:
                logger.warning("CrossExchangeStream connection error: %s", exc)
            if self._running:
                time.sleep(5)

    def _connect(self) -> None:
        products = list(self._product_map.keys())
        url = "wss://ws-feed.exchange.coinbase.com"

        def on_open(ws):
            subscribe_msg = json.dumps({
                "type": "subscribe",
                "product_ids": products,
                "channels": ["ticker"],
            })
            ws.send(subscribe_msg)

        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            data = json.loads(message)
            if data.get("type") != "ticker":
                return
            product_id = data.get("product_id", "")
            symbol = self._product_map.get(product_id)
            if not symbol:
                return
            price = float(data.get("price", 0))
            if price > 0:
                with self._lock:
                    self._coinbase_prices[symbol] = price
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.debug("CrossExchange parse error: %s", exc)

    def _on_error(self, ws: Any, error: Any) -> None:
        logger.debug("CrossExchangeStream WS error: %s", error)

    def _on_close(self, ws: Any, close_status: Any = None, close_msg: Any = None) -> None:
        logger.debug("CrossExchangeStream WS closed")

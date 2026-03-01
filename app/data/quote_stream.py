# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Real-time quote WebSocket stream using alpaca-py.

Provides thread-safe bid/ask/spread/microprice data injected into
market_state for improved limit-order placement and spread penalty.

Streams equities via StockDataStream and crypto via CryptoDataStream.
Falls back gracefully if alpaca-py is not installed.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class QuoteStream:
    """Manages live quote subscriptions for equities and crypto."""

    def __init__(self, api_key: str, api_secret: str, cfg: dict):
        self._api_key = api_key
        self._api_secret = api_secret
        self._feed = str(cfg.get("feed", "iex"))
        self._crypto_enabled = bool(cfg.get("crypto", True))
        self._quotes: dict[str, dict] = {}  # symbol → latest quote
        self._lock = threading.Lock()
        self._equity_stream = None
        self._crypto_stream = None
        self._equity_thread: Optional[threading.Thread] = None
        self._crypto_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, symbols: list[str]) -> None:
        """Start background WebSocket threads for the given symbol list."""
        if self._running:
            self.update_symbols(symbols)
            return
        self._running = True
        equity_syms = [s for s in symbols if "/" not in s]
        crypto_syms = [s for s in symbols if "/" in s]

        if equity_syms:
            self._start_equity_stream(equity_syms)
        if crypto_syms and self._crypto_enabled:
            self._start_crypto_stream(crypto_syms)

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Thread-safe read of the latest quote for a symbol.

        Returns dict with keys: bid, ask, bid_size, ask_size, spread_pct, microprice
        Returns None if no data available yet.
        """
        with self._lock:
            return dict(self._quotes[symbol]) if symbol in self._quotes else None

    def update_symbols(self, symbols: list[str]) -> None:
        """Subscribe to additional symbols (no-op if already subscribed)."""
        if not self._running:
            self.start(symbols)
            return
        equity_new = [s for s in symbols if "/" not in s and s not in self._quotes]
        crypto_new = [s for s in symbols if "/" in s and s not in self._quotes]
        if equity_new and self._equity_stream is not None:
            try:
                self._equity_stream.subscribe_quotes(self._on_equity_quote, *equity_new)
            except Exception as exc:
                logger.debug("QuoteStream equity subscribe failed: %s", exc)
        if crypto_new and self._crypto_stream is not None and self._crypto_enabled:
            try:
                self._crypto_stream.subscribe_quotes(self._on_crypto_quote, *crypto_new)
            except Exception as exc:
                logger.debug("QuoteStream crypto subscribe failed: %s", exc)

    def stop(self) -> None:
        """Stop both WebSocket streams."""
        self._running = False
        for stream in (self._equity_stream, self._crypto_stream):
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _on_equity_quote(self, quote) -> None:
        self._store_quote(quote)

    async def _on_crypto_quote(self, quote) -> None:
        self._store_quote(quote)

    def _store_quote(self, quote) -> None:
        try:
            symbol = str(quote.symbol)
            bid = float(quote.bid_price or 0.0)
            ask = float(quote.ask_price or 0.0)
            bid_size = float(getattr(quote, "bid_size", 0) or 0.0)
            ask_size = float(getattr(quote, "ask_size", 0) or 0.0)
            mid = (bid + ask) / 2.0 if bid and ask else 0.0
            spread_pct = ((ask - bid) / mid * 100.0) if mid > 0 else 0.0
            # Microprice: size-weighted mid
            total_size = bid_size + ask_size
            if total_size > 0:
                microprice = (bid * ask_size + ask * bid_size) / total_size
            else:
                microprice = mid
            with self._lock:
                self._quotes[symbol] = {
                    "bid": bid,
                    "ask": ask,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "spread_pct": round(spread_pct, 4),
                    "microprice": round(microprice, 6),
                }
        except Exception as exc:
            logger.debug("QuoteStream _store_quote error: %s", exc)

    def _start_equity_stream(self, symbols: list[str]) -> None:
        try:
            from alpaca.data.live import StockDataStream  # type: ignore[import]
            self._equity_stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._api_secret,
                feed=self._feed,
            )
            self._equity_stream.subscribe_quotes(self._on_equity_quote, *symbols)
            t = threading.Thread(target=self._equity_stream.run, daemon=True, name="QuoteStreamEquity")
            t.start()
            self._equity_thread = t
            logger.info("QuoteStream equity started (%d symbols)", len(symbols))
        except ImportError:
            logger.warning("alpaca-py not installed — QuoteStream equity unavailable")
        except Exception as exc:
            logger.warning("QuoteStream equity failed to start: %s", exc)

    def _start_crypto_stream(self, symbols: list[str]) -> None:
        try:
            from alpaca.data.live import CryptoDataStream  # type: ignore[import]
            self._crypto_stream = CryptoDataStream(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
            self._crypto_stream.subscribe_quotes(self._on_crypto_quote, *symbols)
            t = threading.Thread(target=self._crypto_stream.run, daemon=True, name="QuoteStreamCrypto")
            t.start()
            self._crypto_thread = t
            logger.info("QuoteStream crypto started (%d symbols)", len(symbols))
        except ImportError:
            logger.warning("alpaca-py not installed — QuoteStream crypto unavailable")
        except Exception as exc:
            logger.warning("QuoteStream crypto failed to start: %s", exc)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Binance Spot broker adapter.

Symbol format: Fricktrade uses "BTC/USD" or "BTC/USDT"; Binance uses "BTCUSDT".
_to_binance_symbol() / _from_binance_symbol() handle the conversion.

Limitations vs Alpaca:
  - No built-in paper mode (use testnet=True with Binance testnet API keys).
  - No avg_entry returned for spot balances (position_state uses None).
  - cancel_order() requires symbol; an in-memory map is maintained per process.
  - PDT rules do not apply (crypto-only broker).
"""

from __future__ import annotations

import logging
from typing import Any

from app.brokers.base import Broker
from app.monitoring.broker_metrics import record_broker_call

# Quote currencies Binance uses (longest first to avoid ambiguous splits)
_BINANCE_QUOTES = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"]


def _to_binance_symbol(symbol: str) -> str:
    """Convert 'BTC/USD' or 'BTC/USDT' → 'BTCUSDT'."""
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        # Treat USD as USDT on Binance
        if quote.upper() == "USD":
            quote = "USDT"
        return f"{base.upper()}{quote.upper()}"
    return symbol.upper()


def _from_binance_symbol(symbol: str) -> str:
    """Convert 'BTCUSDT' → 'BTC/USDT'."""
    sym = symbol.upper()
    for quote in _BINANCE_QUOTES:
        if sym.endswith(quote) and len(sym) > len(quote):
            base = sym[: -len(quote)]
            return f"{base}/{quote}"
    return symbol


class BinanceBroker(Broker):
    """Broker adapter for Binance Spot trading."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        name: str = "binance",
    ):
        try:
            from binance.client import Client  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "python-binance is required for BinanceBroker. "
                "Install it with: pip install python-binance"
            ) from exc

        self._name = name
        self._testnet = testnet
        self.client = Client(api_key, api_secret, testnet=testnet)
        # Maps order_id → Binance symbol string (required for cancel)
        self._order_symbol_map: dict[str, str] = {}

    # --- Broker interface ---

    def is_connected(self) -> bool:
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_account(self) -> dict:
        def _fetch() -> Any:
            return self.client.get_account()

        account = record_broker_call(self._name, "get_account", _fetch)
        usdt_free = 0.0
        usdt_locked = 0.0
        for balance in account.get("balances", []):
            if balance.get("asset") == "USDT":
                usdt_free = float(balance.get("free", 0) or 0)
                usdt_locked = float(balance.get("locked", 0) or 0)
                break
        buying_power = usdt_free
        equity = usdt_free + usdt_locked
        return {
            "equity": equity,
            "cash": usdt_free,
            "buying_power": buying_power,
            # Expose raw balances for downstream use if needed
            "_raw_balances": account.get("balances", []),
        }

    def get_positions(self) -> list[dict]:
        """Return non-USDT spot balances as positions.

        avg_entry is not available from Binance Spot; returns None.
        """
        def _fetch() -> Any:
            return self.client.get_account()

        account = record_broker_call(self._name, "get_positions", _fetch)
        positions: list[dict] = []
        for balance in account.get("balances", []):
            asset = balance.get("asset", "")
            if asset in ("USDT", "BUSD", "USDC"):
                continue
            free = float(balance.get("free", 0) or 0)
            locked = float(balance.get("locked", 0) or 0)
            qty = free + locked
            if qty < 1e-8:
                continue
            positions.append(
                {
                    "symbol": f"{asset}/USDT",
                    "qty": qty,
                    "avg_entry": None,  # not tracked by Binance Spot
                    "side": "long",
                    "asset_class": "crypto",
                }
            )
        return positions

    def get_open_orders(self) -> list[dict]:
        def _fetch() -> Any:
            return self.client.get_open_orders()

        orders = record_broker_call(self._name, "get_open_orders", _fetch)
        result: list[dict] = []
        for order in orders:
            order_id = str(order.get("orderId", ""))
            binance_sym = str(order.get("symbol", ""))
            if order_id:
                self._order_symbol_map[order_id] = binance_sym
            executed_qty = float(order.get("executedQty", 0) or 0)
            cumulative_quote = float(order.get("cummulativeQuoteQty", 0) or 0)
            filled_avg = (
                cumulative_quote / executed_qty if executed_qty > 0 else None
            )
            result.append(
                {
                    "order_id": order_id,
                    "symbol": _from_binance_symbol(binance_sym),
                    "side": str(order.get("side", "")).lower(),
                    "qty": float(order.get("origQty", 0) or 0),
                    "limit_price": float(order.get("price", 0) or 0) or None,
                    "status": str(order.get("status", "")).lower(),
                    "filled_qty": executed_qty,
                    "filled_avg_price": filled_avg,
                }
            )
        return result

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        binance_sym = _to_binance_symbol(symbol)
        side_upper = side.upper()

        def _submit() -> Any:
            if str(order_type).lower() == "limit":
                limit_price = kwargs.get("limit_price")
                if limit_price is None:
                    raise ValueError("limit_price required for limit orders")
                return self.client.order_limit(
                    symbol=binance_sym,
                    side=side_upper,
                    quantity=str(qty),
                    price=str(float(limit_price)),
                    timeInForce="GTC",
                )
            return self.client.order_market(
                symbol=binance_sym,
                side=side_upper,
                quantity=str(qty),
            )

        order = record_broker_call(self._name, "place_order", _submit)
        order_id = str(order.get("orderId", ""))
        if order_id:
            self._order_symbol_map[order_id] = binance_sym
        return order_id

    def close_position(self, symbol: str) -> None:
        """Market-sell the full free balance of the base asset."""
        binance_sym = _to_binance_symbol(symbol)
        base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()

        def _fetch_account() -> Any:
            return self.client.get_account()

        account = record_broker_call(self._name, "close_position", _fetch_account)
        free_qty = 0.0
        for balance in account.get("balances", []):
            if balance.get("asset") == base:
                free_qty = float(balance.get("free", 0) or 0)
                break
        if free_qty < 1e-8:
            return

        def _sell() -> Any:
            return self.client.order_market_sell(
                symbol=binance_sym,
                quantity=str(free_qty),
            )

        try:
            order = record_broker_call(self._name, "close_position_sell", _sell)
            order_id = str(order.get("orderId", ""))
            if order_id:
                self._order_symbol_map[order_id] = binance_sym
        except Exception as exc:
            exc_str = str(exc).lower()
            if "insufficient" in exc_str or "not enough" in exc_str:
                logging.info("Binance close_position %s: no balance to sell.", symbol)
                return
            logging.warning("Failed to close Binance position %s: %s", symbol, exc)
            raise

    def cancel_order(self, order_id: str) -> None:
        if not order_id:
            return
        binance_sym = self._order_symbol_map.get(str(order_id))
        if not binance_sym:
            logging.warning(
                "Binance cancel_order: symbol unknown for order_id=%s "
                "(order may have been placed before this process started)",
                order_id,
            )
            return
        try:
            record_broker_call(
                self._name,
                "cancel_order",
                lambda: self.client.cancel_order(
                    symbol=binance_sym,
                    orderId=int(order_id),
                ),
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "unknown order" in exc_str or "order does not exist" in exc_str:
                return
            logging.warning("Failed to cancel Binance order %s: %s", order_id, exc)

    # --- Overrides ---

    def get_asset_class(self, symbol: str) -> str:
        return "crypto"  # Binance only trades crypto

    def supports_short(self, symbol: str) -> bool:
        return False  # Spot trading only; futures would require a separate adapter

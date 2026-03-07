# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Binance broker adapter — Spot and Futures (FAPI) modes.

Symbol format: Fricktrade uses "BTC/USD" or "BTC/USDT"; Binance uses "BTCUSDT".
_to_binance_symbol() / _from_binance_symbol() handle the conversion.

Spot mode (futures=False):
  - Uses /api/v3/* endpoints.
  - No avg_entry from balances (position_state receives None).
  - cancel_order() requires symbol; in-memory map maintained per process.

Futures mode (futures=True):
  - Uses /fapi/* endpoints (USD-M perpetual contracts).
  - Positions include avg_entry (entryPrice) and unrealized PnL.
  - Supports shorts (positionAmt < 0).
  - Demo trading: set base_url: https://demo-fapi.binance.com
    The client is initialised with demo=True, which routes to FUTURES_DEMO_URL.

PDT rules do not apply (crypto-only broker).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.brokers.base import Broker
from app.monitoring.broker_metrics import record_broker_call

# Quote currencies Binance uses (longest first to avoid ambiguous splits)
_BINANCE_QUOTES = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"]

# Stablecoin assets to skip when building spot positions (balance-based)
_BINANCE_STABLECOINS = frozenset([
    "USDT", "BUSD", "USDC", "DAI", "TUSD", "FDUSD", "USDS", "USDP",
    "PYUSD", "GUSD", "EUR", "EURI", "GBP", "TRY", "BRL", "ARS",
])

# USD-pegged stablecoins counted at 1:1 in equity (subset of _BINANCE_STABLECOINS)
_USD_PEGGED_STABLECOINS = frozenset([
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDS", "USDP", "PYUSD", "GUSD",
])


def _to_binance_symbol(symbol: str) -> str:
    """Convert 'BTC/USD' or 'BTC/USDT' → 'BTCUSDT'."""
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
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


# Process-level cache: binance_symbol → LOT_SIZE stepSize
_LOT_STEP_CACHE: dict[str, float] = {}


def _get_lot_step(client: Any, binance_sym: str) -> float:
    """Return LOT_SIZE stepSize for a Binance symbol (cached per process)."""
    if binance_sym in _LOT_STEP_CACHE:
        return _LOT_STEP_CACHE[binance_sym]
    try:
        info = client.get_symbol_info(binance_sym)
        if info:
            for f in info.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    step = float(f.get("stepSize", 1.0) or 1.0)
                    _LOT_STEP_CACHE[binance_sym] = step
                    return step
    except Exception:
        pass
    return 1.0  # safe default: floor to integer


def _floor_to_step(qty: float, step: float) -> float:
    """Floor qty to the nearest multiple of step."""
    if step <= 0:
        return qty
    factor = 1.0 / step
    return math.floor(qty * factor) / factor


class BinanceBroker(Broker):
    """Broker adapter for Binance Spot or Futures (FAPI) trading."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        futures: bool = False,
        base_url: str = "",
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
        self._futures = futures
        self._base_url = base_url.rstrip("/") if base_url else ""

        # Use the library's built-in demo flag when base_url is a demo endpoint.
        # Client(demo=True) automatically routes to API_DEMO_URL (spot) and
        # FUTURES_DEMO_URL (futures) — no manual URL overrides needed.
        _demo = bool(self._base_url and "demo" in self._base_url.lower())
        self.client = Client(api_key, api_secret, testnet=testnet, demo=_demo)

        _mode = ("futures" if futures else "spot") + (" demo" if _demo else " live")
        logging.info("BinanceBroker(%s): %s", name, _mode)

        # Maps order_id → Binance symbol string (required for cancel)
        self._order_symbol_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _server_ip() -> str:
        """Best-effort fetch of this server's outbound IP for whitelist hints."""
        try:
            import requests as _req
            return _req.get("https://api.ipify.org", timeout=5).text.strip()
        except Exception:
            return "<unknown>"

    def _log_auth_error(self, exc: Exception) -> None:
        exc_str = str(exc)
        if "-2015" in exc_str or "Invalid API-key" in exc_str:
            ip = self._server_ip()
            logging.error(
                "Binance auth failed (code -2015 — IP whitelist or permissions). "
                "Go to Binance → API Management → edit your key and either "
                "add this server's IP (%s) to the whitelist, "
                "or set 'Unrestricted' access. "
                "Also ensure 'Enable Reading' and 'Enable %s Trading' "
                "permissions are checked.",
                ip,
                "Futures" if self._futures else "Spot & Margin",
            )
        else:
            logging.warning("Binance(%s) is_connected() failed: %s", self._name, exc)

    # ------------------------------------------------------------------
    # Broker interface
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Verify connectivity with an authenticated call (not just ping()).

        ping() needs no auth and always succeeds — using it would let
        a bad/IP-restricted key pass startup and spam errors every cycle.
        """
        try:
            if self._futures:
                self.client.futures_account()
            else:
                self.client.get_account()
            return True
        except Exception as exc:
            self._log_auth_error(exc)
            return False

    def get_account(self) -> dict:
        if self._futures:
            return self._futures_account()
        return self._spot_account()

    def get_positions(self) -> list[dict]:
        if self._futures:
            return self._futures_positions()
        return self._spot_positions()

    def get_open_orders(self) -> list[dict]:
        if self._futures:
            return self._futures_open_orders()
        return self._spot_open_orders()

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        if self._futures:
            return self._futures_place_order(symbol, side, qty, order_type, **kwargs)
        return self._spot_place_order(symbol, side, qty, order_type, **kwargs)

    def close_position(self, symbol: str) -> None:
        if self._futures:
            self._futures_close_position(symbol)
        else:
            self._spot_close_position(symbol)

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
            if self._futures:
                record_broker_call(
                    self._name, "cancel_order",
                    lambda: self.client.futures_cancel_order(
                        symbol=binance_sym, orderId=int(order_id)
                    ),
                )
            else:
                record_broker_call(
                    self._name, "cancel_order",
                    lambda: self.client.cancel_order(
                        symbol=binance_sym, orderId=int(order_id)
                    ),
                )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "unknown order" in exc_str or "order does not exist" in exc_str:
                return
            logging.warning("Failed to cancel Binance order %s: %s", order_id, exc)

    # ------------------------------------------------------------------
    # Spot implementations
    # ------------------------------------------------------------------

    def _spot_account(self) -> dict:
        account = record_broker_call(
            self._name, "get_account", self.client.get_account
        )
        usdt_free = usdt_locked = 0.0
        usd_stable_free = usd_stable_locked = 0.0
        non_stable: list[tuple[str, float]] = []
        for balance in account.get("balances", []):
            asset = balance.get("asset", "")
            free = float(balance.get("free", 0) or 0)
            locked = float(balance.get("locked", 0) or 0)
            qty = free + locked
            if qty < 1e-8:
                continue
            if asset in _USD_PEGGED_STABLECOINS:
                usd_stable_free += free
                usd_stable_locked += locked
                if asset == "USDT":
                    usdt_free = free
                    usdt_locked = locked
            elif asset not in _BINANCE_STABLECOINS:
                non_stable.append((asset, qty))

        equity = usd_stable_free + usd_stable_locked
        # Add market value of held crypto assets so equity doesn't appear to
        # drop when USDT is spent buying — this prevents false VaR spikes.
        for asset, qty in non_stable:
            try:
                ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                price = float(ticker.get("price", 0) or 0)
                equity += qty * price
            except Exception:
                pass  # skip if price unavailable
        return {"equity": equity, "cash": usdt_free, "buying_power": usdt_free}

    def _spot_positions(self) -> list[dict]:
        account = record_broker_call(
            self._name, "get_positions", self.client.get_account
        )
        positions: list[dict] = []
        for balance in account.get("balances", []):
            asset = balance.get("asset", "")
            if asset in _BINANCE_STABLECOINS:
                continue
            qty = float(balance.get("free", 0) or 0) + float(balance.get("locked", 0) or 0)
            if qty < 1e-8:
                continue
            positions.append(
                {
                    "symbol": f"{asset}/USD",
                    "qty": qty,
                    "avg_entry": None,  # not tracked by Binance Spot
                    "side": "long",
                    "asset_class": "crypto",
                }
            )
        return positions

    def _spot_open_orders(self) -> list[dict]:
        orders = record_broker_call(
            self._name, "get_open_orders", self.client.get_open_orders
        )
        return self._normalise_orders(orders)

    def _spot_place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        binance_sym = _to_binance_symbol(symbol)
        side_upper = side.upper()
        step = _get_lot_step(self.client, binance_sym)
        adj_qty = _floor_to_step(qty, step)
        if adj_qty <= 0:
            raise ValueError(f"Quantity {qty} floors to 0 for {symbol} (step={step})")

        def _submit() -> Any:
            if str(order_type).lower() == "limit":
                limit_price = kwargs.get("limit_price")
                if limit_price is None:
                    raise ValueError("limit_price required for limit orders")
                return self.client.order_limit(
                    symbol=binance_sym, side=side_upper,
                    quantity=str(adj_qty), price=str(float(limit_price)),
                    timeInForce="GTC",
                )
            return self.client.order_market(
                symbol=binance_sym, side=side_upper, quantity=str(adj_qty)
            )

        order = record_broker_call(self._name, "place_order", _submit)
        return self._register_order(order, binance_sym)

    def _spot_close_position(self, symbol: str) -> None:
        binance_sym = _to_binance_symbol(symbol)
        base = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        account = record_broker_call(
            self._name, "close_position", self.client.get_account
        )
        free_qty = 0.0
        for balance in account.get("balances", []):
            if balance.get("asset") == base:
                free_qty = float(balance.get("free", 0) or 0)
                break
        if free_qty < 1e-8:
            return
        step = _get_lot_step(self.client, binance_sym)
        sell_qty = _floor_to_step(free_qty, step)
        if sell_qty < step:
            # Qty is below LOT_SIZE step — attempt Binance dust conversion to BNB
            logging.info(
                "Binance close_position %s: qty %s below step %s, attempting dust conversion.",
                symbol, free_qty, step,
            )
            try:
                self.client.transfer_dust(asset=[base])
                logging.info("Binance dust conversion succeeded for %s.", base)
            except Exception as exc:
                logging.warning("Binance dust conversion failed for %s: %s", base, exc)
            return
        try:
            order = record_broker_call(
                self._name, "close_position_sell",
                lambda: self.client.order_market_sell(
                    symbol=binance_sym, quantity=str(sell_qty)
                ),
            )
            self._register_order(order, binance_sym)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "insufficient" in exc_str or "not enough" in exc_str:
                logging.info("Binance close_position %s: no balance to sell.", symbol)
                return
            logging.warning("Failed to close Binance spot position %s: %s", symbol, exc)
            raise

    # ------------------------------------------------------------------
    # Futures implementations
    # ------------------------------------------------------------------

    def _futures_account(self) -> dict:
        account = record_broker_call(
            self._name, "get_account", self.client.futures_account
        )
        wallet = float(account.get("totalWalletBalance", 0) or 0)
        unrealised = float(account.get("totalUnrealizedProfit", 0) or 0)
        available = float(account.get("availableBalance", 0) or 0)
        equity = wallet + unrealised
        return {"equity": equity, "cash": available, "buying_power": available}

    def _futures_positions(self) -> list[dict]:
        positions_raw = record_broker_call(
            self._name, "get_positions", self.client.futures_position_information
        )
        positions: list[dict] = []
        for pos in positions_raw:
            amt = float(pos.get("positionAmt", 0) or 0)
            if abs(amt) < 1e-8:
                continue
            binance_sym = str(pos.get("symbol", ""))
            entry = float(pos.get("entryPrice", 0) or 0) or None
            positions.append(
                {
                    "symbol": _from_binance_symbol(binance_sym),
                    "qty": abs(amt),
                    "avg_entry": entry,
                    "side": "long" if amt > 0 else "short",
                    "unrealized_pnl": float(pos.get("unrealizedProfit", 0) or 0),
                    "asset_class": "crypto",
                }
            )
        return positions

    def _futures_open_orders(self) -> list[dict]:
        orders = record_broker_call(
            self._name, "get_open_orders", self.client.futures_get_open_orders
        )
        return self._normalise_orders(orders)

    def _futures_place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        binance_sym = _to_binance_symbol(symbol)
        side_upper = side.upper()
        params: dict[str, Any] = {
            "symbol": binance_sym,
            "side": side_upper,
            "quantity": str(qty),
        }
        if str(order_type).lower() == "limit":
            limit_price = kwargs.get("limit_price")
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            params["type"] = "LIMIT"
            params["price"] = str(float(limit_price))
            params["timeInForce"] = "GTC"
        else:
            params["type"] = "MARKET"

        def _submit() -> Any:
            return self.client.futures_create_order(**params)

        order = record_broker_call(self._name, "place_order", _submit)
        return self._register_order(order, binance_sym)

    def _futures_close_position(self, symbol: str) -> None:
        """Close a futures position with a reduce-only market order."""
        binance_sym = _to_binance_symbol(symbol)
        positions = self._futures_positions()
        qty = 0.0
        close_side = "SELL"
        for pos in positions:
            if pos.get("symbol") == symbol:
                qty = float(pos.get("qty", 0) or 0)
                close_side = "SELL" if pos.get("side") == "long" else "BUY"
                break
        if qty < 1e-8:
            return
        try:
            order = record_broker_call(
                self._name, "close_position",
                lambda: self.client.futures_create_order(
                    symbol=binance_sym,
                    side=close_side,
                    type="MARKET",
                    quantity=str(qty),
                    reduceOnly="true",
                ),
            )
            self._register_order(order, binance_sym)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "position" in exc_str and ("not exist" in exc_str or "no position" in exc_str):
                return
            logging.warning("Failed to close Binance futures position %s: %s", symbol, exc)
            raise

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _normalise_orders(self, orders: list) -> list[dict]:
        result: list[dict] = []
        for order in orders:
            order_id = str(order.get("orderId", ""))
            binance_sym = str(order.get("symbol", ""))
            if order_id:
                self._order_symbol_map[order_id] = binance_sym
            executed_qty = float(order.get("executedQty", 0) or 0)
            cumulative_quote = float(order.get("cumQuote", 0) or order.get("cummulativeQuoteQty", 0) or 0)
            filled_avg = (cumulative_quote / executed_qty) if executed_qty > 0 else None
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

    def _register_order(self, order: dict, binance_sym: str) -> str:
        order_id = str(order.get("orderId", ""))
        if order_id:
            self._order_symbol_map[order_id] = binance_sym
        return order_id

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_asset_class(self, symbol: str) -> str:
        return "crypto"

    def supports_short(self, symbol: str) -> bool:
        return self._futures  # Futures supports short; Spot does not

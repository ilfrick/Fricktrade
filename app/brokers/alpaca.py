# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from app.brokers.base import Broker
from app.monitoring.broker_metrics import record_broker_call


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, api_secret: str, base_url: str, paper: bool = True, name: str = "alpaca"):
        self.client = TradingClient(api_key, api_secret, paper=paper, url_override=base_url)
        self._name = name

    def is_connected(self) -> bool:
        try:
            account_data = self.get_account()
            # If get_account returns data, it means is_success_check passed for get_account
            return bool(account_data)
        except Exception:
            return False

    def get_account(self) -> dict:
        def _check_account_data(account_obj) -> bool:
            if account_obj is None:
                return False
            data = account_obj.dict()
            return bool(data and data.get("status")) # Check if dict is not empty and has a 'status' key

        account = record_broker_call(
            self._name,
            "get_account",
            self.client.get_account,
            is_success_check=_check_account_data,
        )
        data = account.dict()
        if "shorting_enabled" in data:
            data["shorting_enabled"] = bool(data.get("shorting_enabled"))
        return data

    def get_positions(self) -> list[dict]:
        def _fetch_positions():
            try:
                return self.client.get_all_positions()
            except AttributeError:
                return self.client.list_positions()

        def _check_positions_data(positions_list) -> bool:
            return positions_list is not None # Ensure list is not None

        positions = record_broker_call(
            self._name,
            "get_positions",
            _fetch_positions,
            is_success_check=_check_positions_data,
        )
        result = []
        for pos in positions:
            d = pos.dict() if hasattr(pos, "dict") else dict(pos)
            # Alpaca Trading API sometimes returns crypto positions in legacy
            # no-slash format ("BTCUSD" instead of "BTC/USD"). Normalise so
            # position_state keys match trading symbol keys throughout the system.
            asset_class = str(d.get("asset_class", "") or "").lower()
            sym = str(d.get("symbol", "") or "")
            if "crypto" in asset_class and "/" not in sym and sym.upper().endswith("USD"):
                d["symbol"] = sym[:-3] + "/USD"
            result.append(d)
        return result

    def get_open_orders(self) -> list[dict]:
        def _fetch_orders():
            try:
                return self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
            except AttributeError:
                return self.client.list_orders(status="open")

        def _check_orders_data(orders_list) -> bool:
            return orders_list is not None # Ensure list is not None

        orders = record_broker_call(
            self._name,
            "get_open_orders",
            _fetch_orders,
            is_success_check=_check_orders_data,
        )
        results = []
        for order in orders:
            data = order.dict() if hasattr(order, "dict") else dict(order)
            results.append(
                {
                    "order_id": data.get("id") or data.get("order_id"),
                    "symbol": data.get("symbol"),
                    "side": data.get("side"),
                    "qty": float(data.get("qty") or 0.0),
                    "limit_price": data.get("limit_price"),
                    "status": data.get("status"),
                    "filled_qty": float(data.get("filled_qty") or 0.0),
                    "filled_avg_price": data.get("filled_avg_price"),
                }
            )
        return results

    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        is_crypto = "/" in symbol
        # Crypto requires GTC (no DAY orders); equities use DAY by default
        tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY
        extended_hours = bool(kwargs.get("extended_hours", False)) and not is_crypto
        order_kwargs: dict = {}
        if extended_hours:
            order_kwargs["extended_hours"] = True
        if str(order_type).lower() == "limit":
            limit_price = kwargs.get("limit_price")
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            try:
                order_req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=float(limit_price),
                    **order_kwargs,
                )
            except TypeError as exc:
                if order_kwargs and "extended_hours" in str(exc):
                    logging.warning("Extended-hours flag not supported by Alpaca SDK; retrying without it.")
                    order_req = LimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=order_side,
                        time_in_force=tif,
                        limit_price=float(limit_price),
                    )
                else:
                    raise
        else:
            try:
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    **order_kwargs,
                )
            except TypeError as exc:
                if order_kwargs and "extended_hours" in str(exc):
                    logging.warning("Extended-hours flag not supported by Alpaca SDK; retrying without it.")
                    order_req = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=order_side,
                        time_in_force=tif,
                    )
                else:
                    raise
        order = record_broker_call(
            self._name,
            "place_order",
            self.client.submit_order,
            order_req,
        )
        return order.id

    def close_position(self, symbol: str) -> None:
        try:
            record_broker_call(self._name, "close_position", self.client.close_position, symbol)
        except Exception as exc:
            # Only ignore if position does not exist
            exc_str = str(exc).lower()
            if "not found" in exc_str or "no position" in exc_str or "does not exist" in exc_str:
                return
            logging.warning("Failed to close position %s: %s", symbol, exc)
            raise

    def cancel_order(self, order_id: str) -> None:
        if not order_id:
            return
        record_broker_call(self._name, "cancel_order", self.client.cancel_order_by_id, order_id)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import logging

from app.brokers.base import Broker


class ExecutionEngine:
    def __init__(self, broker: Broker):
        self.broker = broker

    def execute(self, symbol: str, action: str, qty: float) -> str | None:
        if action == "buy":
            try:
                return self.broker.place_order(symbol, "buy", qty, "market")
            except Exception as exc:
                logging.warning("Order failed (buy %s qty=%s): %s", symbol, qty, exc)
                return None
        if action == "sell":
            try:
                return self.broker.place_order(symbol, "sell", qty, "market")
            except Exception as exc:
                logging.warning("Order failed (sell %s qty=%s): %s", symbol, qty, exc)
                return None
        if action == "exit":
            self.broker.close_position(symbol)
            return None
        return None

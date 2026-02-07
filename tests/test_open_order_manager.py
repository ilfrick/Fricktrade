# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.agents.open_orders import OpenOrderManager


class TestHasPending:
    def test_empty_cache(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        assert mgr.has_pending("AAPL") is False

    def test_has_pending_true(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [{"symbol": "AAPL", "broker": "alpaca", "side": "buy"}]
        assert mgr.has_pending("AAPL") is True

    def test_has_pending_wrong_symbol(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [{"symbol": "GOOG", "broker": "alpaca", "side": "buy"}]
        assert mgr.has_pending("AAPL") is False

    def test_has_pending_broker_filter(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [{"symbol": "AAPL", "broker": "ibkr", "side": "buy"}]
        assert mgr.has_pending("AAPL", broker="alpaca") is False
        assert mgr.has_pending("AAPL", broker="ibkr") is True

    def test_skip_if_pending_disabled(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {"skip_if_pending": False}}}, "alpaca")
        mgr.cache = [{"symbol": "AAPL", "broker": "alpaca", "side": "buy"}]
        assert mgr.has_pending("AAPL") is False


class TestGetPending:
    def test_get_pending(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [
            {"symbol": "AAPL", "broker": "alpaca", "side": "buy"},
            {"symbol": "GOOG", "broker": "alpaca", "side": "sell"},
            {"symbol": "AAPL", "broker": "ibkr", "side": "sell"},
        ]
        pending = mgr.get_pending("AAPL")
        assert len(pending) == 2
        pending = mgr.get_pending("AAPL", broker="alpaca")
        assert len(pending) == 1


class TestRemovePending:
    def test_remove_all_for_symbol(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [
            {"symbol": "AAPL", "broker": "alpaca"},
            {"symbol": "GOOG", "broker": "alpaca"},
        ]
        mgr.remove_pending("AAPL")
        assert len(mgr.cache) == 1
        assert mgr.cache[0]["symbol"] == "GOOG"

    def test_remove_for_broker(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [
            {"symbol": "AAPL", "broker": "alpaca"},
            {"symbol": "AAPL", "broker": "ibkr"},
        ]
        mgr.remove_pending("AAPL", broker="alpaca")
        assert len(mgr.cache) == 1
        assert mgr.cache[0]["broker"] == "ibkr"


class TestSymbolsForBroker:
    def test_single_broker(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [
            {"symbol": "AAPL", "side": "buy"},
            {"symbol": "GOOG", "side": "sell"},
        ]
        syms = mgr.symbols_for_broker("alpaca", {"alpaca": None})
        assert set(syms) == {"AAPL", "GOOG"}

    def test_multi_broker(self):
        mgr = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
        mgr.cache = [
            {"symbol": "AAPL", "broker": "alpaca", "side": "buy"},
            {"symbol": "GOOG", "broker": "ibkr", "side": "sell"},
        ]
        syms = mgr.symbols_for_broker("alpaca", {"alpaca": None, "ibkr": None})
        assert syms == ["AAPL"]

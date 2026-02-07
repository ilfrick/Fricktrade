# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.agents.symbol_manager import SymbolManager
from app.agents.open_orders import OpenOrderManager


def _make_mgr(cfg=None, broker_map=None, default_broker="alpaca"):
    cfg = cfg or {}
    broker_map = broker_map or {"alpaca": None}
    oom = OpenOrderManager({"execution": {"open_orders": {}}}, default_broker)
    return SymbolManager(cfg, broker_map, default_broker, oom)


class TestResolveActiveSymbols:
    def test_returns_symbols_when_no_strategy(self):
        mgr = _make_mgr()
        mgr.symbols = ["AAPL", "GOOG"]
        assert set(mgr.resolve_active_symbols()) == {"AAPL", "GOOG"}

    def test_merges_strategy_symbols(self):
        mgr = _make_mgr()
        mgr.symbols = ["AAPL"]
        mgr.symbols_by_strategy = {
            "momentum": ["AAPL", "GOOG"],
            "trend": ["MSFT"],
        }
        result = mgr.resolve_active_symbols()
        assert set(result) == {"AAPL", "GOOG", "MSFT"}

    def test_merges_broker_symbols_if_no_strategy(self):
        mgr = _make_mgr()
        mgr.symbols = ["AAPL"]
        mgr.symbols_by_broker = {
            "alpaca": ["AAPL", "GOOG"],
            "ibkr": ["MSFT"],
        }
        result = mgr.resolve_active_symbols()
        assert set(result) == {"AAPL", "GOOG", "MSFT"}


class TestMergeSymbolsWithPositions:
    def test_no_positions(self):
        mgr = _make_mgr()
        result = mgr.merge_symbols_with_positions(["AAPL"], {})
        assert result == ["AAPL"]

    def test_adds_held_positions(self):
        mgr = _make_mgr()
        portfolio = {"positions": {"GOOG": {"qty": 10}, "MSFT": {"qty": 0}}}
        result = mgr.merge_symbols_with_positions(["AAPL"], portfolio)
        assert "AAPL" in result
        assert "GOOG" in result
        # MSFT has qty=0, should not be added
        assert "MSFT" not in result


class TestSymbolVenue:
    def test_from_venues_map(self):
        mgr = _make_mgr()
        mgr.symbol_venues = {"AAPL": "US"}
        assert mgr.symbol_venue("AAPL") == "US"

    def test_default_venue(self):
        mgr = _make_mgr(cfg={"market": {"default_symbol_venue": "US"}})
        assert mgr.symbol_venue("AAPL") == "US"

    def test_none_when_no_config(self):
        mgr = _make_mgr()
        assert mgr.symbol_venue("AAPL") is None


class TestSymbolSector:
    def test_from_config(self):
        mgr = _make_mgr(cfg={"market": {"symbol_sectors": {"AAPL": "tech"}}})
        assert mgr.symbol_sector("AAPL") == "tech"

    def test_none_when_missing(self):
        mgr = _make_mgr()
        assert mgr.symbol_sector("AAPL") is None


class TestMergeWithPositions:
    def test_prioritizes_held_then_open_orders(self):
        mgr = _make_mgr()
        mgr._open_order_mgr.cache = [{"symbol": "TSLA"}]
        portfolio = {"positions": {"GOOG": {"qty": 5}}}
        result = mgr.merge_with_positions(["AAPL", "MSFT"], portfolio, max_symbols=10)
        # Held first, then open orders, then candidates
        assert result[0] == "GOOG"
        assert "TSLA" in result
        assert "AAPL" in result
        assert "MSFT" in result

    def test_respects_max_symbols(self):
        mgr = _make_mgr()
        portfolio = {"positions": {}}
        result = mgr.merge_with_positions(["AAPL", "GOOG", "MSFT"], portfolio, max_symbols=2)
        assert len(result) == 2

    def test_empty_everything(self):
        mgr = _make_mgr()
        result = mgr.merge_with_positions([], {}, max_symbols=10)
        assert result == []


class TestApplyCashCap:
    def test_cash_aware_disabled(self):
        mgr = _make_mgr()
        result = mgr.apply_cash_cap(1.0, 1000.0, {}, {"cash_aware": False})
        assert result == 1000.0

    def test_caps_to_buying_power(self):
        mgr = _make_mgr()
        portfolio = {"cash": 100.0, "buying_power": 500.0, "equity": 1000.0}
        result = mgr.apply_cash_cap(1.0, float("inf"), portfolio, {"cash_aware": True, "cash_max_pct": 100.0, "cash_buffer_pct": 100.0})
        assert result == 500.0

    def test_zero_funds(self):
        mgr = _make_mgr()
        portfolio = {"cash": 0.0, "buying_power": 0.0, "equity": 0.0}
        result = mgr.apply_cash_cap(1.0, 1000.0, portfolio, {"cash_aware": True})
        assert result == 0.0


class TestCapSymbolsByCash:
    def test_disabled(self):
        mgr = _make_mgr()
        result = mgr.cap_symbols_by_cash(50, {}, {"cash_aware": False})
        assert result == 50

    def test_zero_price_min(self):
        mgr = _make_mgr()
        result = mgr.cap_symbols_by_cash(50, {}, {"cash_aware": True, "filters": {"price_min": 0}})
        assert result == 50

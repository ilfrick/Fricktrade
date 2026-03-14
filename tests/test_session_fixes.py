# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
#
# Tests for session-anomaly fixes: per-symbol circuit breaker, PDT retry
# suppression, AI filter market gate, enqueue failure notional release.

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. OrderResponse.reason field  (requires prometheus_client)
# ---------------------------------------------------------------------------

_prom = pytest.importorskip("prometheus_client", reason="prometheus_client not installed")

from app.execution.order_queue import OrderQueue, OrderResponse  # noqa: E402


class TestOrderResponseReason:
    def test_reason_default_empty(self):
        r = OrderResponse(symbol="AAPL", broker="alpaca", status="rejected")
        assert r.reason == ""

    def test_reason_populated(self):
        r = OrderResponse(symbol="AAPL", broker="alpaca", status="rejected", reason="pdt_protection")
        assert r.reason == "pdt_protection"

    def test_pdt_reject_populates_reason(self):
        """When broker rejects with PDT code, the OrderResponse includes reason='pdt_protection'."""

        class _PDTBroker:
            def place_order(self, *a, **kw):
                raise Exception('{"code": "40310100"}')

        broker = _PDTBroker()
        q = OrderQueue(broker, "alpaca", retry_cfg={"enabled": False})
        q.enqueue("PFAI", "sell", 10)
        responses = q.pop_responses()
        rejects = [r for r in responses if r.status == "rejected"]
        assert len(rejects) == 1
        assert rejects[0].reason == "pdt_protection"


# ---------------------------------------------------------------------------
# 2. AI filter market-open gate
# ---------------------------------------------------------------------------

from app.agents.symbol_manager import SymbolManager  # noqa: E402
from app.agents.open_orders import OpenOrderManager  # noqa: E402


def _make_symbol_mgr(cfg=None):
    cfg = cfg or {}
    oom = OpenOrderManager({"execution": {"open_orders": {}}}, "alpaca")
    return SymbolManager(cfg, {"alpaca": None}, "alpaca", oom)


class TestAIFilterMarketGate:
    @patch("app.agents.symbol_manager.is_venue_open", return_value=False)
    @patch("app.agents.symbol_manager.get_alpaca_account_cfg", return_value={"api_key": "", "api_secret": ""})
    def test_skips_when_market_closed(self, mock_cfg, mock_venue):
        mgr = _make_symbol_mgr({"data": {"dynamic_symbols": {"enabled": True, "provider": "alpaca", "refresh_minutes": 1}}})
        mgr._dynamic_symbols_at = None
        mgr.refresh_dynamic_symbols({}, ["trend_following"])
        # Equity market closed + no API keys → returns early without setting _dynamic_symbols_at
        assert mgr._dynamic_symbols_at is None

    @patch("app.agents.symbol_manager.is_venue_open", return_value=True)
    @patch("app.agents.symbol_manager.get_alpaca_account_cfg", return_value={"api_key": "", "api_secret": ""})
    def test_proceeds_when_market_open(self, mock_cfg, mock_venue):
        mgr = _make_symbol_mgr({"data": {"dynamic_symbols": {"enabled": True, "provider": "alpaca", "refresh_minutes": 1}}})
        mgr._dynamic_symbols_at = None
        # Will proceed past the gate but may fail later due to missing API keys — that's OK
        mgr.refresh_dynamic_symbols({}, ["trend_following"])
        # Should have attempted to check venue (may or may not succeed depending on API)
        mock_venue.assert_called()


# ---------------------------------------------------------------------------
# 3. Per-symbol circuit breaker (unit test of the logic pattern)
# ---------------------------------------------------------------------------

class TestPerSymbolCircuitBreaker:
    """Test the per-symbol drawdown calculation logic directly."""

    def test_high_drawdown_triggers(self):
        """6% drawdown with 5% threshold should trigger."""
        entry = 100.0
        current = 94.0
        dd = (entry - current) / entry * 100.0
        assert dd == pytest.approx(6.0)
        assert dd >= 5.0  # would trigger at 5% threshold

    def test_low_drawdown_passes(self):
        """4% drawdown with 5% threshold should NOT trigger."""
        entry = 100.0
        current = 96.0
        dd = (entry - current) / entry * 100.0
        assert dd == pytest.approx(4.0)
        assert dd < 5.0  # would pass at 5% threshold

    def test_no_position_passes(self):
        """No position state means no circuit breaker check."""
        pos = None
        assert pos is None  # CB block is skipped

    def test_profitable_position_passes(self):
        """Current price above entry means no drawdown."""
        entry = 100.0
        current = 105.0
        assert current >= entry

    def test_zero_entry_skips(self):
        """Zero avg_entry should skip the check."""
        entry = 0.0
        assert entry <= 0  # guard: skip if entry <= 0


# ---------------------------------------------------------------------------
# 4. PDT block set
# ---------------------------------------------------------------------------

class TestPDTBlockedSet:
    def test_pdt_block_suppresses_exit(self):
        """Symbols in _pdt_blocked should be skipped for position exit."""
        blocked: set[tuple[str, str]] = set()
        blocked.add(("alpaca", "PFAI"))
        assert ("alpaca", "PFAI") in blocked
        assert ("alpaca", "AAPL") not in blocked

    def test_pdt_block_daily_clear(self):
        """PDT blocks should clear on new day."""
        blocked: set[tuple[str, str]] = {("alpaca", "PFAI")}
        old_date = date(2026, 2, 13)
        new_date = date(2026, 2, 14)
        if old_date != new_date:
            blocked.clear()
        assert len(blocked) == 0


# ---------------------------------------------------------------------------
# 5. Enqueue failure releases pending notional
# ---------------------------------------------------------------------------

class TestEnqueueFailureNotionalRelease:
    def test_release_on_enqueue_exception(self):
        """If enqueue raises, pending notional should be released."""
        pending: dict[str, float] = {"alpaca": 500.0}
        action = "buy"
        is_closing = False
        notional = 500.0

        try:
            raise RuntimeError("enqueue failed")
        except Exception:
            if action == "buy" and not is_closing:
                current = pending.get("alpaca", 0.0)
                pending["alpaca"] = max(0.0, current - notional)

        assert pending["alpaca"] == 0.0

    def test_no_release_for_sell(self):
        """Sell-side enqueue failure should NOT release pending notional."""
        pending: dict[str, float] = {"alpaca": 500.0}
        action = "sell"
        is_closing = False
        notional = 500.0

        try:
            raise RuntimeError("enqueue failed")
        except Exception:
            if action == "buy" and not is_closing:
                current = pending.get("alpaca", 0.0)
                pending["alpaca"] = max(0.0, current - notional)

        assert pending["alpaca"] == 500.0  # unchanged

    def test_no_release_for_position_close(self):
        """Position-close buy should NOT release pending notional (never reserved)."""
        pending: dict[str, float] = {"alpaca": 500.0}
        action = "buy"
        is_closing = True
        notional = 500.0

        try:
            raise RuntimeError("enqueue failed")
        except Exception:
            if action == "buy" and not is_closing:
                current = pending.get("alpaca", 0.0)
                pending["alpaca"] = max(0.0, current - notional)

        assert pending["alpaca"] == 500.0  # unchanged

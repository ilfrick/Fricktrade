# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Tests for TradingAgent._size_order logic."""

import pytest

pytest.importorskip("tensorflow")
from unittest.mock import MagicMock, patch

from app.agents.trader import TradingAgent


def _make_agent(
    max_pos_pct=5.0,
    max_short_pct=5.0,
    allow_shorts=False,
    vol_targeting=None,
):
    broker = MagicMock()
    broker.get_account.return_value = {"equity": 100000, "cash": 50000, "buying_power": 200000}
    broker.get_positions.return_value = []
    broker.get_open_orders.return_value = []
    broker.is_connected.return_value = True
    cfg = {
        "strategy": {
            "name": "intraday_momentum",
            "names": ["intraday_momentum"],
            "params": {"allow_shorts": allow_shorts},
        },
        "risk": {
            "max_position_size_pct": max_pos_pct,
            "max_short_exposure_pct": max_short_pct,
            "max_daily_loss_pct": 2.0,
            "max_portfolio_leverage": 2.0,
            "circuit_breaker_drawdown_pct": 10.0,
        },
        "data": {},
        "execution": {},
        "market": {},
    }
    if vol_targeting:
        cfg["risk"]["vol_targeting"] = vol_targeting
    return TradingAgent(broker, cfg)


class TestSizeOrderBuy:
    def test_basic_buy(self):
        agent = _make_agent(max_pos_pct=10.0)
        portfolio = {"equity": 100000.0, "cash": 50000.0, "positions": {}}
        # Half-Kelly: kelly_win_prob=1.0 → kelly_scale=0.5 → effective 5% of 100k=$5k → qty=50
        # tod_scale patched to 1.0 to avoid time-of-day flakiness
        with patch.object(type(agent), "_time_of_day_scale", staticmethod(lambda s, a: 1.0)):
            qty, reason = agent._size_order("buy", 100.0, portfolio, "AAPL", {"kelly_win_prob": 1.0})
        assert qty > 0
        assert reason is None
        assert qty == 50

    def test_buy_zero_price(self):
        agent = _make_agent()
        qty, reason = agent._size_order("buy", 0.0, {}, "AAPL", {})
        assert qty == 0
        assert reason == "no_price"

    def test_buy_no_cash(self):
        agent = _make_agent()
        portfolio = {"equity": 100000.0, "cash": 0.0, "positions": {}}
        qty, reason = agent._size_order("buy", 100.0, portfolio, "AAPL", {})
        assert qty == 0
        assert reason == "insufficient_cash"

    def test_buy_no_equity(self):
        agent = _make_agent()
        portfolio = {"equity": 0.0, "cash": 1000.0, "positions": {}}
        qty, reason = agent._size_order("buy", 100.0, portfolio, "AAPL", {})
        assert qty == 0
        assert reason == "insufficient_cash"

    def test_buy_existing_position(self):
        agent = _make_agent(max_pos_pct=10.0)
        portfolio = {
            "equity": 100000.0,
            "cash": 50000.0,
            "positions": {"AAPL": {"qty": 25}},
        }
        # kelly_win_prob=1.0 → kelly_scale=0.5 → effective 5% of 100k=$5k target
        # existing 25*100=$2.5k, remaining $2.5k → qty=25
        with patch.object(type(agent), "_time_of_day_scale", staticmethod(lambda s, a: 1.0)):
            qty, reason = agent._size_order("buy", 100.0, portfolio, "AAPL", {"kelly_win_prob": 1.0})
        assert qty == 25
        assert reason is None

    def test_buy_position_limit(self):
        agent = _make_agent(max_pos_pct=5.0)
        portfolio = {
            "equity": 100000.0,
            "cash": 50000.0,
            "positions": {"AAPL": {"qty": 60}},
        }
        # Target 5k, existing 60*100=6k > 5k, remaining=0
        qty, reason = agent._size_order("buy", 100.0, portfolio, "AAPL", {})
        assert qty == 0
        assert reason == "position_limit"

    def test_buy_cash_limited(self):
        agent = _make_agent(max_pos_pct=50.0)
        portfolio = {"equity": 100000.0, "cash": 1000.0, "positions": {}}
        # Target 50k but only 1000 cash, price=100, qty=10
        qty, reason = agent._size_order("buy", 100.0, portfolio, "AAPL", {})
        assert qty == 10
        assert reason is None


class TestSizeOrderSell:
    def test_sell_existing_position(self):
        agent = _make_agent()
        portfolio = {
            "equity": 100000.0,
            "cash": 50000.0,
            "positions": {"AAPL": {"qty": 100}},
        }
        qty, reason = agent._size_order("sell", 100.0, portfolio, "AAPL", {})
        assert qty == 100
        assert reason is None

    def test_sell_reduce_pct(self):
        agent = _make_agent()
        portfolio = {
            "equity": 100000.0,
            "cash": 50000.0,
            "positions": {"AAPL": {"qty": 100}},
        }
        qty, reason = agent._size_order("sell", 100.0, portfolio, "AAPL", {}, reduce_pct=0.5)
        assert qty == 50
        assert reason is None

    def test_sell_no_position_shorts_disabled(self):
        agent = _make_agent(allow_shorts=False)
        portfolio = {"equity": 100000.0, "cash": 50000.0, "positions": {}}
        qty, reason = agent._size_order("sell", 100.0, portfolio, "AAPL", {})
        assert qty == 0
        assert reason == "shorting_disabled"

    def test_sell_zero_position(self):
        agent = _make_agent()
        portfolio = {
            "equity": 100000.0,
            "cash": 50000.0,
            "positions": {"AAPL": {"qty": 0}},
        }
        qty, reason = agent._size_order("sell", 100.0, portfolio, "AAPL", {})
        assert qty == 0
        assert reason == "shorting_disabled"


class TestSizeOrderUnsupported:
    def test_unsupported_action(self):
        agent = _make_agent()
        portfolio = {"equity": 100000.0, "cash": 50000.0, "positions": {}}
        qty, reason = agent._size_order("unknown", 100.0, portfolio, "AAPL", {})
        assert qty == 0
        assert reason == "unsupported"

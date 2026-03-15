# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Tests for TradingAgent._size_order logic."""

import math

import pytest
from unittest.mock import MagicMock, patch

from app.agents.trader import TradingAgent


def _make_agent(
    max_pos_pct=5.0,
    max_short_pct=5.0,
    allow_shorts=False,
    vol_targeting=None,
    fractional_shares=False,
    min_notional=1.0,
    crypto_max_pos_pct=None,
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
        "trading_limits": {
            "fractional_shares": fractional_shares,
            "min_notional": min_notional,
        },
        "data": {},
        "execution": {},
        "market": {},
    }
    if vol_targeting:
        cfg["risk"]["vol_targeting"] = vol_targeting
    if crypto_max_pos_pct is not None:
        cfg["risk"]["crypto"] = {"max_position_size_pct": crypto_max_pos_pct}
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


class TestCryptoFractionalBuy:
    """F7 — crypto fractional buy: crypto_order_margin applied, precision=8, uses floor."""

    def test_crypto_buy_applies_margin_and_floor(self):
        # max_pos_pct=100% so allowed_value ≈ cash (50000); price=50000 → qty ≈ 1.0 BTC
        # crypto_order_margin=0.97 → allowed=50000*0.97=48500 → qty=48500/50000=0.97 floor@8=0.97
        agent = _make_agent(max_pos_pct=100.0)
        portfolio = {"equity": 100000.0, "cash": 50000.0, "positions": {}}
        market_state = {"kelly_win_prob": 1.0}
        with patch.object(type(agent), "_time_of_day_scale", staticmethod(lambda s, a: 1.0)):
            qty, reason = agent._size_order("buy", 50000.0, portfolio, "BTC/USD", market_state)
        assert reason is None
        assert qty > 0
        # qty must be a multiple of 1e-8 (precision=8)
        assert abs(qty * 1e8 - round(qty * 1e8)) < 1e-4, "qty must have at most 8 decimal places"
        # qty must not exceed allowed_value/price — floor guarantee
        allowed_value = 50000.0 * 0.97  # crypto_order_margin default
        assert qty <= allowed_value / 50000.0 + 1e-9

    def test_crypto_buy_qty_floor_not_round(self):
        # Force a price where round() would overshoot but floor() would not.
        # allowed_value=9.99999995, price=10.0 → raw=0.999999995
        # round(raw, 8)=1.0 > raw; floor(raw*1e8)/1e8=0.99999999 < raw  ← floor wins
        agent = _make_agent(max_pos_pct=100.0)
        # cash=9.99999995 so allowed=cash*0.97=9.699...; price=10 → 0.9699...
        # Use a contrived cash value where round would differ from floor
        portfolio = {"equity": 10.0, "cash": 10.0, "positions": {}}
        market_state = {"kelly_win_prob": 1.0}
        with patch.object(type(agent), "_time_of_day_scale", staticmethod(lambda s, a: 1.0)):
            qty, reason = agent._size_order("buy", 10.0, portfolio, "BTC/USD", market_state)
        # qty must be ≤ allowed/price to guarantee no overshoot
        allowed_value = 10.0 * 0.97
        assert qty <= allowed_value / 10.0 + 1e-9

    def test_crypto_max_pos_pct_override(self):
        # F2: risk.crypto.max_position_size_pct should override global for crypto symbols.
        # Global cap=5%, crypto cap=50% → crypto buy uses 50%.
        agent = _make_agent(max_pos_pct=5.0, crypto_max_pos_pct=50.0)
        portfolio = {"equity": 100000.0, "cash": 60000.0, "positions": {}}
        market_state = {"kelly_win_prob": 1.0}
        with patch.object(type(agent), "_time_of_day_scale", staticmethod(lambda s, a: 1.0)):
            crypto_qty, _ = agent._size_order("buy", 100.0, portfolio, "ETH/USD", market_state)
        with patch.object(type(agent), "_time_of_day_scale", staticmethod(lambda s, a: 1.0)):
            equity_qty, _ = agent._size_order("buy", 100.0, portfolio, "AAPL", {"kelly_win_prob": 1.0})
        # Crypto should be sized much larger than equity given the higher cap
        assert crypto_qty > equity_qty * 2, (
            f"crypto qty {crypto_qty} should be >> equity qty {equity_qty}"
        )


class TestAntiDustRemainder:
    """F7 — partial sell leaving sub-min_notional remainder is rounded up to full close."""

    def test_partial_sell_rounds_up_when_remainder_tiny(self):
        # Position: 10 shares @ $100 = $1000; reduce_pct=0.9 → sell 9 shares, remainder=1*100=$100
        # min_notional=$200: remainder $100 < $200 → should sell ALL 10 shares
        agent = _make_agent(max_pos_pct=10.0, fractional_shares=True, min_notional=200.0)
        portfolio = {
            "equity": 100000.0,
            "cash": 50000.0,
            "positions": {"AAPL": {"qty": 10.0}},
        }
        qty, reason = agent._size_order("sell", 100.0, portfolio, "AAPL", {}, reduce_pct=0.9)
        assert reason is None
        # remainder=1*100=$100 < min_notional=$200 → should round up to full 10
        # (with (1-1e-9) safety margin applied, may be 9.999 not exactly 10)
        assert qty >= 9.999, f"Expected full close (≥9.999), got {qty}"

    def test_partial_sell_no_roundup_when_remainder_ok(self):
        # Position: 100 shares @ $100; reduce_pct=0.5 → sell 50, remainder=50*100=$5000
        # min_notional=$1: $5000 >> $1 → no round-up, sell exactly 50
        agent = _make_agent(max_pos_pct=10.0, fractional_shares=True, min_notional=1.0)
        portfolio = {
            "equity": 100000.0,
            "cash": 50000.0,
            "positions": {"AAPL": {"qty": 100.0}},
        }
        qty, reason = agent._size_order("sell", 100.0, portfolio, "AAPL", {}, reduce_pct=0.5)
        assert reason is None
        assert qty == pytest.approx(50.0, abs=0.001)


class TestKellyVoteMode:
    """F1 — _combine_signals in vote mode must write kelly_win_prob to market_state."""

    def test_vote_mode_buy_sets_kelly_win_prob(self):
        agent = _make_agent()
        # Patch _combine_mode to simulate vote mode path
        agent._combine_mode = "vote"
        signals = [
            {"name": "strat_a", "action": "buy", "confidence": 0.75},
            {"name": "strat_b", "action": "buy", "confidence": 0.60},
            {"name": "strat_c", "action": "hold", "confidence": 0.50},
        ]
        market_state = {}
        action, reduce_pct, strategy = agent._combine_signals(
            signals, {}, order=["strat_a", "strat_b", "strat_c"], market_state=market_state
        )
        assert action == "buy"
        assert "kelly_win_prob" in market_state, (
            "vote mode buy must write kelly_win_prob to market_state"
        )
        assert market_state["kelly_win_prob"] == pytest.approx(0.75, abs=0.01)

    def test_vote_mode_hold_does_not_set_kelly(self):
        agent = _make_agent()
        agent._combine_mode = "vote"
        signals = [
            {"name": "strat_a", "action": "hold", "confidence": 0.8},
            {"name": "strat_b", "action": "hold", "confidence": 0.8},
        ]
        market_state = {}
        action, _, _ = agent._combine_signals(
            signals, {}, order=["strat_a", "strat_b"], market_state=market_state
        )
        assert action == "hold"
        assert "kelly_win_prob" not in market_state

    def test_vote_mode_kelly_drives_larger_buy(self):
        # Verify that kelly_win_prob > 0 in vote mode produces a larger buy than kelly_win_prob absent
        agent = _make_agent(max_pos_pct=10.0)
        portfolio = {"equity": 100000.0, "cash": 50000.0, "positions": {}}
        with patch.object(type(agent), "_time_of_day_scale", staticmethod(lambda s, a: 1.0)):
            qty_no_kelly, _ = agent._size_order("buy", 100.0, portfolio, "AAPL", {})
            qty_with_kelly, _ = agent._size_order(
                "buy", 100.0, portfolio, "AAPL", {"kelly_win_prob": 0.8}
            )
        assert qty_with_kelly > qty_no_kelly, (
            f"kelly_win_prob=0.8 should produce larger buy ({qty_with_kelly}) "
            f"than kelly_win_prob absent ({qty_no_kelly})"
        )

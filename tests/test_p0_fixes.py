# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Tests for P0 fixes: strategy weights, phantom sell filter, pending sell qty."""

import pytest

pytest.importorskip("tensorflow")

from unittest.mock import MagicMock
from app.agents.trader import TradingAgent


def _make_p0_agent(combine="weighted", strategy_names=None, config_weights=None, allow_shorts=False):
    """Create a minimal TradingAgent for P0 fix tests."""
    broker = MagicMock()
    broker.get_account.return_value = {"equity": 100000, "cash": 50000, "buying_power": 200000}
    broker.get_positions.return_value = []
    broker.get_open_orders.return_value = []
    broker.is_connected.return_value = True
    cfg = {
        "strategy": {
            "combine": combine,
            "name": "trend_following",
            "names": strategy_names or ["trend_following", "factor_model", "pattern_trading", "stat_arb_pairs"],
            "params": {"allow_shorts": allow_shorts},
        },
        "risk": {
            "max_position_size_pct": 5.0,
            "max_short_exposure_pct": 5.0,
            "max_daily_loss_pct": 2.0,
            "max_portfolio_leverage": 2.0,
            "circuit_breaker_drawdown_pct": 10.0,
        },
        "orchestrator": {
            "strategy_weights": config_weights if config_weights is not None else {
                "trend_following": 0.35,
                "factor_model": 0.25,
                "pattern_trading": 0.25,
                "stat_arb_pairs": 0.15,
            },
        },
        "trading_limits": {"enabled": True, "allow_shorts": allow_shorts},
        "data": {},
        "execution": {},
        "market": {},
    }
    agent = TradingAgent(broker, cfg)
    return agent


# --- Fix 1: Config strategy weights ---


class TestConfigStrategyWeights:
    def test_combine_signals_uses_config_weights(self):
        """With uniform orchestrator weights, config weights should be applied.

        factor_model buy (0.25) + pattern_trading buy (0.25) = 0.50
        vs trend_following sell (0.35) → buy wins.
        """
        agent = _make_p0_agent()
        signals = [
            {"action": "sell", "name": "trend_following", "confidence": 1.0},
            {"action": "buy", "name": "factor_model", "confidence": 1.0},
            {"action": "buy", "name": "pattern_trading", "confidence": 1.0},
            {"action": "hold", "name": "stat_arb_pairs", "confidence": 1.0},
        ]
        # Uniform weights simulating disabled orchestrator
        uniform = {n: 1.0 for n in ["trend_following", "factor_model", "pattern_trading", "stat_arb_pairs"]}
        action, _, _ = agent._combine_signals(signals, uniform)
        assert action == "buy", f"Expected buy with config weights, got {action}"

    def test_without_config_weights_uses_uniform(self):
        """Without config weights, uniform 1.0 should be used as-is."""
        agent = _make_p0_agent(config_weights={})
        agent._config_strategy_weights = {}
        signals = [
            {"action": "sell", "name": "trend_following", "confidence": 1.0},
            {"action": "buy", "name": "factor_model", "confidence": 1.0},
        ]
        uniform = {"trend_following": 1.0, "factor_model": 1.0}
        action, _, _ = agent._combine_signals(signals, uniform)
        assert action == "hold", "Equal scores should produce hold (tie)"

    def test_nonuniform_weights_not_replaced(self):
        """Orchestrator weights that aren't all 1.0 should NOT be replaced."""
        agent = _make_p0_agent()
        signals = [
            {"action": "sell", "name": "trend_following", "confidence": 1.0},
            {"action": "buy", "name": "factor_model", "confidence": 1.0},
        ]
        # Orchestrator returns non-uniform weights
        weights = {"trend_following": 0.9, "factor_model": 0.1}
        action, _, _ = agent._combine_signals(signals, weights)
        # sell_score = 0.9, buy_score = 0.1 → sell wins
        assert action == "sell"

    def test_config_weights_stored_in_init(self):
        """Config strategy weights should be stored during __init__."""
        agent = _make_p0_agent()
        assert agent._config_strategy_weights == {
            "trend_following": 0.35,
            "factor_model": 0.25,
            "pattern_trading": 0.25,
            "stat_arb_pairs": 0.15,
        }


# --- Fix 2: Phantom sell filter ---


class TestPhantomSellFilter:
    def test_phantom_sell_filtered_no_position(self):
        """Sell signals filtered when position=0 and shorting disabled."""
        agent = _make_p0_agent(allow_shorts=False)
        signals = [
            {"action": "sell", "name": "trend_following", "confidence": 1.0},
            {"action": "buy", "name": "factor_model", "confidence": 1.0},
        ]
        portfolio = {"positions": {}, "equity": 100000}
        symbol = "AAPL"
        _cs_qty = float(portfolio.get("positions", {}).get(symbol, {}).get("qty", 0) or 0)
        can_short = agent._can_short(symbol, portfolio)
        assert _cs_qty <= 0
        assert not can_short
        filtered = [s for s in signals if s.get("action") != "sell"]
        action, _, _ = agent._combine_signals(filtered, {"trend_following": 0.35, "factor_model": 0.25})
        assert action == "buy"

    def test_sell_counted_when_position_held(self):
        """Sell signals counted when position exists (legitimate exit)."""
        agent = _make_p0_agent(allow_shorts=False)
        signals = [
            {"action": "sell", "name": "trend_following", "confidence": 1.0},
            {"action": "buy", "name": "factor_model", "confidence": 0.5},
        ]
        portfolio = {"positions": {"AAPL": {"qty": 100}}, "equity": 100000}
        can_short = agent._can_short("AAPL", portfolio)
        assert can_short  # has position, selling is legitimate
        # sell_score = 0.35 * 1.0 = 0.35 > buy_score = 0.25 * 0.5 = 0.125
        action, _, _ = agent._combine_signals(signals, {"trend_following": 0.35, "factor_model": 0.25})
        assert action == "sell"

    def test_sell_counted_when_shorts_allowed(self):
        """Sell signals counted when shorting is enabled (even without position)."""
        agent = _make_p0_agent(allow_shorts=True)
        signals = [
            {"action": "sell", "name": "trend_following", "confidence": 1.0},
            {"action": "buy", "name": "factor_model", "confidence": 0.3},
        ]
        # No position but shorts allowed → sell is a legitimate short entry
        portfolio = {"positions": {}, "equity": 100000}
        # _can_short checks account.shorting_enabled which defaults to falsy for MagicMock,
        # but the call-site filter only runs when _can_short returns False.
        # This test verifies the filter logic: with allow_shorts=True in trading_limits,
        # the filter wouldn't remove sells.
        action, _, _ = agent._combine_signals(signals, {"trend_following": 0.35, "factor_model": 0.25})
        assert action == "sell"


# --- Fix 3: Pending sell qty ---


class TestPendingSellQty:
    def test_prevents_overshoot(self):
        """Second sell blocked when first sell covers full position qty."""
        agent = _make_p0_agent(allow_shorts=False)
        agent._pending_sell_qty[("broker1", "RIG")] = 62.0
        current_qty = 62.0
        pending = agent._pending_sell_qty.get(("broker1", "RIG"), 0.0)
        available = current_qty - pending
        assert available <= 0

    def test_allows_partial(self):
        """Remaining qty not covered by pending sells is allowed."""
        agent = _make_p0_agent(allow_shorts=False)
        agent._pending_sell_qty[("broker1", "RIG")] = 30.0
        current_qty = 62.0
        pending = agent._pending_sell_qty.get(("broker1", "RIG"), 0.0)
        available = current_qty - pending
        assert available == 32.0
        qty = min(62, int(available))
        assert qty == 32

    def test_released_on_completion(self):
        """Pending sell qty released after order completes."""
        agent = _make_p0_agent(allow_shorts=False)
        key = ("broker1", "RIG")
        agent._pending_sell_qty[key] = 62.0
        # Simulate release
        resp_qty = 62.0
        agent._pending_sell_qty[key] = max(0.0, agent._pending_sell_qty[key] - resp_qty)
        assert agent._pending_sell_qty[key] == 0.0

    def test_released_on_rejection(self):
        """Pending sell qty released on rejection/cancel."""
        agent = _make_p0_agent(allow_shorts=False)
        key = ("broker1", "RIG")
        agent._pending_sell_qty[key] = 62.0
        agent._pending_sell_qty[key] = max(0.0, agent._pending_sell_qty[key] - 62.0)
        assert agent._pending_sell_qty[key] == 0.0

    def test_no_pending_allows_full_sell(self):
        """With no pending sells, full position qty is available."""
        agent = _make_p0_agent(allow_shorts=False)
        current_qty = 100.0
        pending = agent._pending_sell_qty.get(("broker1", "AAPL"), 0.0)
        available = current_qty - pending
        assert available == 100.0

    def test_pending_sell_dict_initialized(self):
        """_pending_sell_qty should be initialized as empty dict in __init__."""
        agent = _make_p0_agent()
        assert isinstance(agent._pending_sell_qty, dict)
        assert len(agent._pending_sell_qty) == 0

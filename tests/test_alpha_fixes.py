# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Tests for the 2026-08-19 alpha/exit fixes:

1. compute_all_indicators emits "rsi" and "atr" (both were missing, which left
   the ATR stop dead and the peak-detection exit's RSI leg pinned at 50).
2. _vol_target_scale compares a daily-equivalent realized vol against the daily
   target (was per-bar vol vs daily target → always pinned at max_scale).
3. ATR stop scales per-bar ATR to daily-equivalent before applying the mult.
4. RiskManager.check_crypto_exposure counts positions stored under "value".
"""

import math

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.learning.indicators import compute_all_indicators
from app.risk.manager import RiskManager


def _ohlcv(n=200, drift=0.0, noise=0.001, seed=7):
    rng = np.random.default_rng(seed)
    rets = drift + noise * rng.standard_normal(n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    highs = closes * (1.0 + noise)
    lows = closes * (1.0 - noise)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.full(n, 1000.0)
    return opens, highs, lows, closes, volumes


def _make_agent(cfg_overrides=None):
    from app.agents.trader import TradingAgent
    broker = MagicMock()
    broker.get_account.return_value = {"equity": 100000, "cash": 50000}
    broker.get_positions.return_value = []
    broker.brokers = {"alpaca": broker}
    cfg = {
        "strategies": {"active": ["trend_following"], "combine": "first", "params": {}},
        "risk": {
            "enabled": True,
            "vol_targeting": {"enabled": True, "target_vol_pct": 4.0, "min_scale": 0.3, "max_scale": 1.5},
        },
        "data": {"interval": "1m"},
        "learning": {},
        "orchestrator": {"mode": "select", "rl": {"enabled": False}},
        "brokers": {},
    }
    for key, val in (cfg_overrides or {}).items():
        cfg[key] = val
    with patch("app.agents.trader.build_market_cache_config"), \
         patch("app.agents.trader.build_market_cache"):
        agent = TradingAgent(broker, cfg)
    return agent


# --- Fix 1: indicator keys ---------------------------------------------------

class TestIndicatorKeys:
    def test_rsi_and_atr_present(self):
        o, h, l, c, v = _ohlcv()
        ind = compute_all_indicators(o, h, l, c, v)
        assert "rsi" in ind and "atr" in ind
        assert 0.0 <= ind["rsi"] <= 100.0
        assert ind["atr"] > 0.0

    def test_rsi_extremes(self):
        n = 100
        up = np.linspace(100, 200, n)
        o, h, l, v = up, up * 1.001, up * 0.999, np.full(n, 1.0)
        ind = compute_all_indicators(o, h, l, up, v)
        assert ind["rsi"] > 80.0
        down = np.linspace(200, 100, n)
        ind = compute_all_indicators(down, down * 1.001, down * 0.999, down, v)
        assert ind["rsi"] < 20.0


# --- Fix 2: vol targeting units ----------------------------------------------

class TestVolTargetScale:
    def _prices(self, per_bar_vol, n=2000, seed=3):
        rng = np.random.default_rng(seed)
        rets = per_bar_vol * rng.standard_normal(n)
        return (100.0 * np.cumprod(1.0 + rets)).tolist()

    def test_calm_symbol_scales_up(self):
        # 0.02% per-1m-bar vol → ~0.76% daily → well under 4% target → max_scale
        agent = _make_agent()
        ms = {"symbol": "BTC/USD", "prices": self._prices(0.0002)}
        assert agent._vol_target_scale(ms) == pytest.approx(1.5)

    def test_volatile_symbol_scales_down(self):
        # 0.3% per-1m-bar vol → ~11.4% daily → scale ≈ 0.35, NOT pinned at 1.5.
        # This is the regression the fix addresses: the old code compared the
        # per-bar 0.3% against the 4% daily target and returned max_scale.
        agent = _make_agent()
        ms = {"symbol": "DOGE/USD", "prices": self._prices(0.003)}
        scale = agent._vol_target_scale(ms)
        assert scale < 1.0
        assert scale >= 0.3

    def test_disabled_returns_one(self):
        agent = _make_agent({"risk": {"enabled": True, "vol_targeting": {"enabled": False}}})
        ms = {"symbol": "BTC/USD", "prices": self._prices(0.003)}
        assert agent._vol_target_scale(ms) == 1.0


# --- Fix 3: ATR stop daily scaling -------------------------------------------

class TestAtrStopScaling:
    def _exit(self, agent, atr, last_price, avg_entry=100.0):
        from datetime import datetime, timedelta, timezone
        state = agent._broker_state(agent._broker_name)
        state.position_state["BTC/USD"] = {
            "qty": 1.0,
            "avg_entry": avg_entry,
            "opened_at": datetime.now(timezone.utc) - timedelta(hours=2),
            "peak_price": avg_entry,
        }
        ms = {"indicators": {"atr": atr}, "symbol": "BTC/USD"}
        return agent._check_position_exit("BTC/USD", last_price, state, market_state=ms)

    def test_per_bar_atr_does_not_stop_on_noise(self):
        # Per-bar ATR of 0.1 (0.1% of price): unscaled 2.5×ATR would put the stop
        # at 99.75 and a routine -0.5% wiggle would exit. Daily scaling widens it.
        agent = _make_agent({"risk": {
            "enabled": True,
            "crypto": {"hard_stop_pct": 6.0, "atr_stop_mult": 2.5},
            "vol_targeting": {"enabled": False},
        }})
        should_exit, reason = self._exit(agent, atr=0.1, last_price=99.5)
        assert not (should_exit and reason == "atr_stop"), (
            f"0.5% dip must not trigger a daily-scaled ATR stop (got {reason})"
        )

    def test_hard_stop_floor_still_caps_loss(self):
        agent = _make_agent({"risk": {
            "enabled": True,
            "crypto": {"hard_stop_pct": 6.0, "atr_stop_mult": 2.5},
            "vol_targeting": {"enabled": False},
        }})
        # Price below the 6% hard-stop floor always exits regardless of ATR width
        should_exit, reason = self._exit(agent, atr=0.1, last_price=93.0)
        assert should_exit
        assert reason in ("atr_stop", "hard_stop")

    def test_calm_regime_atr_stop_tighter_than_hard_stop(self):
        # ATR 0.03% per bar → daily ≈ 1.14% → 2.5× = 2.85% stop, tighter than 6%.
        agent = _make_agent({"risk": {
            "enabled": True,
            "crypto": {"hard_stop_pct": 6.0, "atr_stop_mult": 2.5},
            "vol_targeting": {"enabled": False},
        }})
        should_exit, reason = self._exit(agent, atr=0.03, last_price=96.5)
        assert should_exit and reason == "atr_stop"


# --- Fix 4: crypto exposure cap key ------------------------------------------

class TestCryptoExposureValueKey:
    def test_existing_value_positions_count(self):
        portfolio = {
            "equity": 10000.0,
            "positions": {
                "BTC/USD": {"qty": 0.05, "value": 4000.0},   # snapshot key
                "ETH/USD": {"qty": 1.0, "market_value": 500.0},  # broker key
            },
        }
        cfg = {"max_crypto_exposure_pct": 50.0, "max_position_size_pct": 25.0}
        # 45% held + 10% new order = 55% > 50% cap → must block
        blocked, reason = RiskManager.check_crypto_exposure(
            "SOL/USD", "buy", 10.0, 100.0, portfolio, cfg,
        )
        assert blocked
        assert "crypto_exposure_cap" in (reason or "")

    def test_per_position_cap_counts_held_value(self):
        portfolio = {
            "equity": 10000.0,
            "positions": {"BTC/USD": {"qty": 0.05, "value": 2000.0}},
        }
        cfg = {"max_crypto_exposure_pct": 90.0, "max_position_size_pct": 25.0}
        # 20% held in BTC + 10% more = 30% > 25% per-position cap → block
        blocked, reason = RiskManager.check_crypto_exposure(
            "BTC/USD", "buy", 10.0, 100.0, portfolio, cfg,
        )
        assert blocked
        assert "crypto_position_cap" in (reason or "")

    def test_small_order_passes(self):
        portfolio = {
            "equity": 10000.0,
            "positions": {"BTC/USD": {"qty": 0.01, "value": 500.0}},
        }
        cfg = {"max_crypto_exposure_pct": 50.0, "max_position_size_pct": 25.0}
        blocked, _ = RiskManager.check_crypto_exposure(
            "ETH/USD", "buy", 1.0, 100.0, portfolio, cfg,
        )
        assert not blocked

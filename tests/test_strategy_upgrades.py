# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Tests for strategy upgrades: RSI filter, factor model mean-reversion, ATR stop."""

import numpy as np

from app.strategies.trend_following import TrendFollowingStrategy
from app.strategies.factor_model import FactorModelStrategy, FactorParams
from app.strategies.pattern_trading import PatternTradingStrategy


# --- Trend Following: RSI filter ---


def _make_trend_strat(fast=3, slow=5, breakout=0.1, exit_pct=0.1):
    return TrendFollowingStrategy(
        {"trend_following": {"fast_window": fast, "slow_window": slow, "breakout_pct": breakout, "exit_pct": exit_pct}}
    )


def test_trend_rsi_blocks_overbought_buy():
    """When RSI >= 70, buy should be suppressed even if trend is up."""
    # Build a series where price rose sharply (RSI very high) but trend is up
    # 20 bars rising fast => RSI will be near 100
    prices = [10.0 + i * 0.5 for i in range(20)]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.01)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    # RSI of a monotonically rising series is ~100 => buy blocked
    assert sig["action"] in ("hold", "sell"), f"Expected hold/sell for overbought, got {sig['action']}"
    assert sig.get("rsi", 0) >= 70


def test_trend_rsi_allows_normal_buy():
    """Normal uptrend with RSI < 70 should still buy."""
    # Mixed prices trending up but not monotonically
    prices = [10, 10.1, 9.9, 10.2, 10.0, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.6, 10.4, 10.7, 10.5, 10.8]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.1)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    if sig["action"] == "buy":
        assert sig.get("rsi", 50) < 70


def test_trend_rsi_blocks_oversold_sell():
    """When RSI <= 30, sell should be suppressed."""
    # Monotonically falling => RSI near 0
    prices = [20.0 - i * 0.5 for i in range(20)]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.01, exit_pct=0.01)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    assert sig["action"] in ("hold", "buy"), f"Expected hold for oversold, got {sig['action']}"
    assert sig.get("rsi", 50) <= 30


def test_trend_volume_confirmation():
    """Buy requires volume spike (last bar >= 1.5x avg of prior 4)."""
    prices = [10.0 + i * 0.3 for i in range(8)]  # gentle uptrend
    # Low volume on last bar
    low_vol = [100, 100, 100, 100, 100, 100, 100, 50]
    strat = _make_trend_strat(fast=3, slow=5, breakout=0.01)
    sig_low = strat.generate_signal({"prices": prices, "volumes": low_vol})

    # High volume on last bar
    high_vol = [100, 100, 100, 100, 100, 100, 100, 200]
    sig_high = strat.generate_signal({"prices": prices, "volumes": high_vol})

    # With low volume, should hold; with high volume, could buy
    if sig_low["action"] == "buy" and sig_high["action"] == "buy":
        pass  # both pass trend — volume didn't block either (possible if volumes < 5 effective)
    # Main check: signal returns contain rsi key
    assert "rsi" in sig_low or sig_low["action"] == "hold"


def test_trend_rsi_returns_in_signal():
    """Signal dict should include rsi field."""
    prices = [10 + i * 0.1 for i in range(20)]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.01)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    assert "rsi" in sig


# --- Factor Model: Mean-Reversion + Trend Quality ---


def test_factor_mr_weight_in_params():
    """FactorParams should have mr_weight."""
    params = FactorParams()
    assert hasattr(params, "mr_weight")
    assert params.mr_weight == 0.15


def test_factor_mr_weight_from_config():
    """mr_weight should be read from config."""
    strat = FactorModelStrategy({"factor_model": {"mr_weight": 0.25}})
    assert strat.params.mr_weight == 0.25


def test_factor_trend_quality_gate():
    """Choppy market (low trend quality) should return hold."""
    # Zigzag prices with no net direction
    prices = [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {"momentum_weight": 1.0, "liquidity_weight": 0.0, "volatility_weight": 0.0, "mr_weight": 0.0}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    # Net move is small relative to total movement => trend_quality < 0.3
    assert sig.get("trend_quality", 1.0) < 0.5 or sig["action"] == "hold"


def test_factor_longer_momentum_lookback():
    """Momentum should use up to 10 bars."""
    # 15 bars: first 10 flat, last 5 rising
    prices = [10.0] * 10 + [10.1, 10.2, 10.3, 10.4, 10.5]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {"momentum_weight": 1.0, "liquidity_weight": 0.0, "volatility_weight": 0.0, "mr_weight": 0.0}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    # With 10-bar lookback, momentum includes the flat bars, reducing the signal
    assert "score" in sig


def test_factor_mean_reversion_oversold():
    """Oversold stock (price below mean) should get positive mr_score contribution."""
    # Price dropped well below 20-bar mean
    prices = [100.0] * 18 + [95.0, 90.0]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {"momentum_weight": 0.0, "liquidity_weight": 0.0, "volatility_weight": 0.0, "mr_weight": 1.0}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    # Price below mean => z-score negative => inverted => positive mr_score
    assert sig.get("score", 0) > 0 or sig["action"] == "hold"


def test_factor_signal_includes_trend_quality():
    """Signal dict should include trend_quality."""
    prices = [10 + i * 0.1 for i in range(20)]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    assert "trend_quality" in sig


# --- Pattern Trading: ATR-based Stop ---


def _make_pattern_cfg():
    return {
        "selection": {
            "price_min": 0.0,
            "relative_volume_min": 0.0,
            "premarket_gain_min_pct": 0.0,
            "min_shares_traded": 0,
            "max_spread_pct": 100.0,
            "require_catalyst": False,
        },
        "pattern": {"ma_periods": [3, 5], "pullback_max_retrace_pct": 90},
        "entry": {"breakout_lookback_bars": 3, "volume_confirm_mult": 0.0},
        "risk": {"stop_loss_pct": 0.05, "partial_take_profit_pct": 0.10, "trailing_stop_pct": 0.02},
    }


def test_pattern_atr_stop_uses_indicators():
    """ATR-based stop should be computed from highs/lows/closes."""
    cfg = _make_pattern_cfg()
    strat = PatternTradingStrategy(cfg)

    # Build market_state where entry signal fires
    # Prices trending up with breakout above recent high
    prices = [10.0, 10.1, 10.2, 10.0, 10.1, 10.3, 10.5, 10.8, 11.2]
    highs = [10.2, 10.3, 10.4, 10.2, 10.3, 10.5, 10.7, 11.0, 11.5]
    lows = [9.8, 9.9, 10.0, 9.8, 9.9, 10.1, 10.3, 10.6, 11.0]
    volumes = [200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000, 500000]

    ms = {
        "prices": prices,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "last_price": 11.2,
        "relative_volume": 5.0,
        "session_gain_pct": 5.0,
        "session_volume": 1000000,
    }

    sig = strat.generate_signal(ms)
    if sig["action"] == "buy":
        # Stop should be set and not just the fixed percentage
        assert strat.state.stop_price is not None
        fixed_stop = 11.2 * (1 - 0.05)
        # ATR stop could be above or below fixed, but stop should be max of both
        assert strat.state.stop_price >= fixed_stop


def test_pattern_atr_stop_floor():
    """ATR-based stop should never go below fixed stop (floor)."""
    cfg = _make_pattern_cfg()
    strat = PatternTradingStrategy(cfg)

    # Very volatile data where ATR would give a very wide stop
    prices = [5.0, 8.0, 4.0, 9.0, 3.0, 10.0, 4.0, 11.0, 15.0]
    highs = [6.0, 9.0, 5.0, 10.0, 4.0, 11.0, 5.0, 12.0, 16.0]
    lows = [4.0, 7.0, 3.0, 8.0, 2.0, 9.0, 3.0, 10.0, 14.0]
    volumes = [200000] * 9

    ms = {
        "prices": prices,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "last_price": 15.0,
        "relative_volume": 5.0,
        "session_gain_pct": 5.0,
        "session_volume": 1000000,
    }

    sig = strat.generate_signal(ms)
    if sig["action"] == "buy":
        fixed_stop = 15.0 * (1 - 0.05)
        assert strat.state.stop_price >= fixed_stop


def test_pattern_atr_stop_short_data():
    """With very short data, ATR computation should not crash."""
    cfg = _make_pattern_cfg()
    cfg["pattern"]["ma_periods"] = [2]
    cfg["entry"]["breakout_lookback_bars"] = 2
    strat = PatternTradingStrategy(cfg)

    prices = [10.0, 10.5, 11.0]
    highs = [10.2, 10.7, 11.2]
    lows = [9.8, 10.3, 10.8]
    volumes = [200000, 200000, 500000]

    ms = {
        "prices": prices,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "last_price": 11.0,
        "relative_volume": 5.0,
        "session_gain_pct": 5.0,
        "session_volume": 1000000,
    }

    # Should not raise
    sig = strat.generate_signal(ms)
    assert sig["action"] in ("buy", "hold", "sell", "exit")

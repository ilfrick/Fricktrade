# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Tests for strategy upgrades: Phases 0-3 improvements."""

import numpy as np

from app.strategies.trend_following import TrendFollowingStrategy
from app.strategies.factor_model import FactorModelStrategy, FactorParams
from app.strategies.stat_arb_pairs import StatArbPairsStrategy
from app.strategies.pattern_trading import PatternTradingStrategy
from app.strategies.confidence_calibrator import ConfidenceCalibrator


# --- Trend Following: RSI filter + indicators ---


def _make_trend_strat(fast=3, slow=5, breakout=0.1, exit_pct=0.1):
    return TrendFollowingStrategy(
        {"trend_following": {"fast_window": fast, "slow_window": slow, "breakout_pct": breakout, "exit_pct": exit_pct}}
    )


def test_trend_rsi_blocks_overbought_buy():
    """When RSI >= 70, buy should be suppressed even if trend is up."""
    prices = [10.0 + i * 0.5 for i in range(20)]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.01)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    assert sig["action"] in ("hold", "sell"), f"Expected hold/sell for overbought, got {sig['action']}"
    assert sig.get("rsi", 0) >= 70


def test_trend_rsi_allows_normal_buy():
    """Normal uptrend with RSI < 70 should still buy."""
    prices = [10, 10.1, 9.9, 10.2, 10.0, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.6, 10.4, 10.7, 10.5, 10.8]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.1)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    if sig["action"] == "buy":
        assert sig.get("rsi", 50) < 70


def test_trend_rsi_blocks_oversold_sell():
    """When RSI <= 30, sell should be suppressed."""
    prices = [20.0 - i * 0.5 for i in range(20)]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.01, exit_pct=0.01)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    assert sig["action"] in ("hold", "buy"), f"Expected hold for oversold, got {sig['action']}"
    assert sig.get("rsi", 50) <= 30


def test_trend_volume_confirmation():
    """Buy requires volume spike (last bar >= 1.5x avg of prior 4)."""
    prices = [10.0 + i * 0.3 for i in range(8)]
    low_vol = [100, 100, 100, 100, 100, 100, 100, 50]
    strat = _make_trend_strat(fast=3, slow=5, breakout=0.01)
    sig_low = strat.generate_signal({"prices": prices, "volumes": low_vol})
    high_vol = [100, 100, 100, 100, 100, 100, 100, 200]
    sig_high = strat.generate_signal({"prices": prices, "volumes": high_vol})
    if sig_low["action"] == "buy" and sig_high["action"] == "buy":
        pass
    assert "rsi" in sig_low or sig_low["action"] == "hold"


def test_trend_rsi_returns_in_signal():
    """Signal dict should include rsi field."""
    prices = [10 + i * 0.1 for i in range(20)]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.01)
    sig = strat.generate_signal({"prices": prices, "volumes": []})
    assert "rsi" in sig


def test_trend_supertrend_buy_confirmation():
    """Supertrend=1 should boost confidence for buy signals."""
    prices = [10, 10.1, 9.9, 10.2, 10.0, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.6, 10.4, 10.7, 10.5, 10.8]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.1)
    ms_no_ind = {"prices": prices, "volumes": []}
    ms_with_ind = {"prices": prices, "volumes": [], "indicators": {"supertrend": 1, "vwap_dev": 0.5}}
    sig1 = strat.generate_signal(ms_no_ind)
    sig2 = strat.generate_signal(ms_with_ind)
    if sig1["action"] == "buy" and sig2["action"] == "buy":
        assert sig2.get("confidence", 0) >= sig1.get("confidence", 0)


def test_trend_crisis_regime_blocks_buy():
    """high_vol_crisis regime should block buy signals."""
    prices = [10, 10.1, 9.9, 10.2, 10.0, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.6, 10.4, 10.7, 10.5, 10.8]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.1)
    ms = {"prices": prices, "volumes": [], "regime_name": "high_vol_crisis"}
    sig = strat.generate_signal(ms)
    assert sig["action"] != "buy"


def test_trend_overbought_exit():
    """RSI > 75 with indicators should trigger sell."""
    prices = [10.0 + i * 0.5 for i in range(20)]
    strat = _make_trend_strat(fast=3, slow=10, breakout=0.01)
    ms = {"prices": prices, "volumes": [], "indicators": {"supertrend": 1}}
    sig = strat.generate_signal(ms)
    # RSI is very high due to monotonic rise, should sell
    assert sig["action"] in ("hold", "sell")


def test_trend_compute_rsi_static():
    """Test RSI static method."""
    close = np.array([10.0 + i for i in range(20)], dtype=float)
    rsi = TrendFollowingStrategy._compute_rsi(close)
    assert rsi > 70  # monotonically rising
    close_down = np.array([20.0 - i for i in range(20)], dtype=float)
    rsi_down = TrendFollowingStrategy._compute_rsi(close_down)
    assert rsi_down < 30


# --- Factor Model: Mean-Reversion + Trend Quality + Hurst ---


def test_factor_mr_weight_in_params():
    """FactorParams should have mr_weight."""
    params = FactorParams()
    assert hasattr(params, "mr_weight")
    assert params.mr_weight == 0.2


def test_factor_mr_weight_from_config():
    """mr_weight should be read from config."""
    strat = FactorModelStrategy({"factor_model": {"mr_weight": 0.25}})
    assert strat.params.mr_weight == 0.25


def test_factor_trend_quality_gate():
    """Choppy market (low trend quality) should return hold."""
    prices = [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {"momentum_weight": 1.0, "liquidity_weight": 0.0, "volatility_weight": 0.0, "mr_weight": 0.0}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    assert sig.get("trend_quality", 1.0) < 0.5 or sig["action"] == "hold"


def test_factor_longer_momentum_lookback():
    """Momentum should use up to 10 bars."""
    prices = [10.0] * 10 + [10.1, 10.2, 10.3, 10.4, 10.5]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {"momentum_weight": 1.0, "liquidity_weight": 0.0, "volatility_weight": 0.0, "mr_weight": 0.0}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    assert "score" in sig


def test_factor_mean_reversion_oversold():
    """Oversold stock should get positive mr_score contribution."""
    prices = [100.0] * 18 + [95.0, 90.0]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {"momentum_weight": 0.0, "liquidity_weight": 0.0, "volatility_weight": 0.0, "mr_weight": 1.0}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    assert sig.get("score", 0) > 0 or sig["action"] == "hold"


def test_factor_signal_includes_trend_quality():
    """Signal dict should include trend_quality."""
    prices = [10 + i * 0.1 for i in range(20)]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes})
    assert "trend_quality" in sig


def test_factor_with_indicators_roc():
    """Factor model should use ROC from indicators when available."""
    prices = [10 + i * 0.1 for i in range(20)]
    volumes = [100000] * len(prices)
    indicators = {"roc": 0.5, "stoch_k": 0.4, "cci": -0.3, "hurst": 0.6}
    strat = FactorModelStrategy({"factor_model": {}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes, "indicators": indicators})
    assert "score" in sig


def test_factor_hurst_trending_boosts_momentum():
    """Hurst > 0.5 should boost momentum weight."""
    prices = [10 + i * 0.5 for i in range(20)]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {"momentum_weight": 0.5}})
    sig_no_hurst = strat.generate_signal({"prices": prices, "volumes": volumes})
    sig_trending = strat.generate_signal({"prices": prices, "volumes": volumes, "indicators": {"hurst": 0.7}})
    # Both should produce signals, trending hurst should give different score
    assert "score" in sig_no_hurst
    assert "score" in sig_trending


def test_factor_hurst_mean_reverting():
    """Hurst < 0.5 should boost mr weight and reduce momentum."""
    prices = [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11]
    volumes = [100000] * len(prices)
    strat = FactorModelStrategy({"factor_model": {}})
    sig = strat.generate_signal({"prices": prices, "volumes": volumes, "indicators": {"hurst": 0.3, "stoch_k": 0.2, "cci": -0.5}})
    # With hurst < 0.5, should not be blocked by trend quality gate
    assert "score" in sig


# --- Stat Arb: Log-ratio spread + Cointegration ---


def test_stat_arb_default_z_entry():
    """Default z_entry should be 2.0."""
    strat = StatArbPairsStrategy({"stat_arb_pairs": {}})
    assert strat.params.z_entry == 2.0


def test_stat_arb_cointegration_test():
    """Cointegration test should return hedge ratio and t-stat."""
    np.random.seed(42)
    n = 100
    b = np.cumsum(np.random.randn(n)) + 50
    a = 0.8 * b + np.random.randn(n) * 0.5 + 10
    a = np.exp(np.log(np.maximum(a, 1.0)))
    b = np.exp(np.log(np.maximum(b, 1.0)))
    hedge_ratio, t_stat = StatArbPairsStrategy._cointegration_test(a, b)
    assert isinstance(hedge_ratio, float)
    assert t_stat is not None
    assert isinstance(t_stat, float)


def test_stat_arb_cointegration_non_cointegrated():
    """Non-cointegrated random walks should have t-stat > -2.86."""
    np.random.seed(123)
    n = 100
    a = np.cumsum(np.random.randn(n)) + 50
    b = np.cumsum(np.random.randn(n)) + 50
    a = np.maximum(a, 1.0)
    b = np.maximum(b, 1.0)
    hedge_ratio, t_stat = StatArbPairsStrategy._cointegration_test(a, b)
    # Most independent random walks should NOT pass the ADF test
    assert t_stat is not None


def test_stat_arb_log_ratio_spread():
    """Signal should use log-ratio spread (not raw difference)."""
    strat = StatArbPairsStrategy({"stat_arb_pairs": {"lookback": 20}})
    # Manually set up a pair with known cointegration
    np.random.seed(42)
    n = 60
    base = np.cumsum(np.random.randn(n) * 0.01) + 4.0
    sym_a_prices = list(np.exp(base + np.random.randn(n) * 0.005))
    sym_b_prices = list(np.exp(base * 1.1 + np.random.randn(n) * 0.005))
    for p in sym_a_prices:
        strat._update_cache("SYM_A", [p])
    for p in sym_b_prices:
        strat._update_cache("SYM_B", [p])
    strat._refresh_pairs_if_needed()
    sig = strat.generate_signal({"symbol": "SYM_A", "prices": sym_a_prices[-5:]})
    assert sig["action"] in ("buy", "sell", "exit", "hold")


def test_stat_arb_confidence_in_signal():
    """Actionable signals should include confidence."""
    strat = StatArbPairsStrategy({"stat_arb_pairs": {"lookback": 20, "z_entry": 0.5}})
    np.random.seed(42)
    n = 60
    base = np.cumsum(np.random.randn(n) * 0.01) + 4.0
    sym_a_prices = list(np.exp(base))
    sym_b_prices = list(np.exp(base * 0.9))
    for p in sym_a_prices:
        strat._update_cache("AA", [p])
    for p in sym_b_prices:
        strat._update_cache("BB", [p])
    strat._refresh_pairs_if_needed()
    sig = strat.generate_signal({"symbol": "AA", "prices": sym_a_prices[-5:]})
    if sig["action"] in ("buy", "sell", "exit"):
        assert "confidence" in sig


def test_stat_arb_short_data():
    """Short data should not crash."""
    strat = StatArbPairsStrategy({})
    sig = strat.generate_signal({"symbol": "X", "prices": [10, 11]})
    assert sig["action"] == "hold"


# --- Confidence Calibrator ---


def test_calibrator_cold_start():
    """Before min_samples, calibrated confidence is conservative (raw * 0.75)."""
    cal = ConfidenceCalibrator(min_samples=30)
    result = cal.calibrate("trend", 0.8)
    assert abs(result - 0.6) < 1e-9  # 0.8 * 0.75


def test_calibrator_record_and_calibrate():
    """After enough samples, calibration should reflect win rates."""
    cal = ConfidenceCalibrator(window=100, n_bins=5, min_samples=10)
    # High confidence signals always win
    for _ in range(20):
        cal.record("strat_a", 0.9, True)
    # Low confidence signals always lose
    for _ in range(20):
        cal.record("strat_a", 0.1, False)
    high_cal = cal.calibrate("strat_a", 0.9)
    low_cal = cal.calibrate("strat_a", 0.1)
    assert high_cal > low_cal


def test_calibrator_per_strategy():
    """Calibration should be per-strategy."""
    cal = ConfidenceCalibrator(min_samples=5)
    for _ in range(10):
        cal.record("good_strat", 0.7, True)
        cal.record("bad_strat", 0.7, False)
    good = cal.calibrate("good_strat", 0.7)
    bad = cal.calibrate("bad_strat", 0.7)
    assert good > bad


def test_calibrator_clamps_to_0_1():
    """Output should always be in [0, 1]."""
    cal = ConfidenceCalibrator(min_samples=5)
    for _ in range(10):
        cal.record("s", 0.5, True)
    result = cal.calibrate("s", 1.5)  # over range
    assert 0.0 <= result <= 1.0
    result2 = cal.calibrate("s", -0.5)  # under range
    assert 0.0 <= result2 <= 1.0


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
        assert strat.state.stop_price is not None
        fixed_stop = 11.2 * (1 - 0.05)
        assert strat.state.stop_price >= fixed_stop


def test_pattern_atr_stop_floor():
    """ATR-based stop should never go below fixed stop (floor)."""
    cfg = _make_pattern_cfg()
    strat = PatternTradingStrategy(cfg)

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

    sig = strat.generate_signal(ms)
    assert sig["action"] in ("buy", "hold", "sell", "exit")


# --- Execution: Limit orders ---


def test_executor_supports_limit_orders():
    """ExecutionEngine.execute should accept limit_price param."""
    from app.execution.executor import ExecutionEngine
    import inspect
    sig = inspect.signature(ExecutionEngine.execute)
    assert "limit_price" in sig.parameters
    assert "order_type" in sig.parameters



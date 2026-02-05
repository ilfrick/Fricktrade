# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Market regime detection for adaptive strategy selection.

Classifies market conditions into volatility regimes, trend regimes, and
provides features for regime-aware trading.
"""

from __future__ import annotations

import numpy as np


def classify_volatility_regime(returns: np.ndarray, window: int = 60) -> int:
    """
    Classify volatility regime based on rolling percentile.

    Returns:
        0 = low volatility (bottom 33%)
        1 = normal volatility (middle 33%)
        2 = high volatility (top 33%)
    """
    if returns.size < window:
        return 1  # default to normal

    current_vol = float(np.std(returns[-window:]))

    # Use historical percentile of volatility
    vols = []
    for i in range(window, returns.size):
        vols.append(np.std(returns[i - window:i]))

    if not vols:
        return 1

    percentile = np.searchsorted(np.sort(vols), current_vol) / len(vols)

    if percentile < 0.33:
        return 0  # low vol
    elif percentile > 0.67:
        return 2  # high vol
    return 1  # normal


def classify_trend_regime(closes: np.ndarray, window: int = 100) -> int:
    """
    Classify trend regime using efficiency ratio and ADX-like measure.

    Returns:
        0 = mean-reverting / choppy
        1 = no clear trend (random walk)
        2 = trending
    """
    if closes.size < window:
        return 1  # default to neutral

    window_data = closes[-window:]

    # Efficiency ratio: net change / total path
    net_change = abs(window_data[-1] - window_data[0])
    total_path = float(np.sum(np.abs(np.diff(window_data))))

    if total_path < 1e-10:
        return 1

    efficiency = net_change / total_path

    # Hurst-like measure from variance ratios
    short_returns = np.diff(window_data[::2])  # skip every other
    long_returns = np.diff(window_data)

    if short_returns.size < 2 or long_returns.size < 2:
        return 1

    var_ratio = np.var(long_returns) / (2 * np.var(short_returns) + 1e-10)

    # Combine signals
    if efficiency > 0.6 and var_ratio > 0.8:
        return 2  # trending
    elif efficiency < 0.3 and var_ratio < 0.5:
        return 0  # mean-reverting
    return 1  # neutral


def classify_correlation_regime(spy_returns: np.ndarray, asset_returns: np.ndarray, window: int = 60) -> int:
    """
    Classify correlation regime (risk-on vs risk-off).

    Returns:
        0 = low correlation (diversification works)
        1 = normal correlation
        2 = high correlation (risk-off, everything moves together)
    """
    if spy_returns.size < window or asset_returns.size < window:
        return 1

    corr = np.corrcoef(spy_returns[-window:], asset_returns[-window:])[0, 1]

    if np.isnan(corr):
        return 1

    if corr < 0.3:
        return 0  # low correlation
    elif corr > 0.7:
        return 2  # high correlation
    return 1


def classify_liquidity_regime(volumes: np.ndarray, spreads: np.ndarray | None, window: int = 60) -> int:
    """
    Classify liquidity regime.

    Returns:
        0 = high liquidity (easy to trade)
        1 = normal liquidity
        2 = low liquidity / stressed
    """
    if volumes.size < window:
        return 1

    current_vol = float(np.mean(volumes[-20:]))
    avg_vol = float(np.mean(volumes[-window:]))

    if avg_vol == 0:
        return 1

    vol_ratio = current_vol / avg_vol

    # Check spread if available
    spread_stressed = False
    if spreads is not None and spreads.size >= 20:
        current_spread = float(np.mean(spreads[-20:]))
        avg_spread = float(np.mean(spreads))
        spread_stressed = current_spread > 2 * avg_spread

    if vol_ratio > 1.5 and not spread_stressed:
        return 0  # high liquidity
    elif vol_ratio < 0.5 or spread_stressed:
        return 2  # low liquidity
    return 1


def compute_regime_features(
    closes: np.ndarray,
    volumes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    spreads: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute regime features as a numpy array.

    Returns 8-element feature vector:
    - volatility_regime (0, 1, 2)
    - trend_regime (0, 1, 2)
    - liquidity_regime (0, 1, 2)
    - volatility_percentile (0-1)
    - trend_strength (-1 to 1)
    - hurst_estimate (0-1)
    - volume_ratio
    - realized_vol
    """
    if closes.size < 10:
        return np.array([1, 1, 1, 0.5, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)

    returns = np.diff(np.log(np.maximum(closes, 1e-10)))

    # Regime classifications
    vol_regime = classify_volatility_regime(returns)
    trend_regime = classify_trend_regime(closes)
    liq_regime = classify_liquidity_regime(volumes, spreads)

    # Continuous features
    window = min(60, closes.size - 1)

    # Volatility percentile
    current_vol = float(np.std(returns[-window:])) if window > 1 else 0.0
    vol_percentile = 0.5
    if returns.size >= window * 2:
        vols = [np.std(returns[i - window:i]) for i in range(window, returns.size)]
        if vols:
            vol_percentile = np.searchsorted(np.sort(vols), current_vol) / len(vols)

    # Trend strength (efficiency ratio, signed)
    if closes.size >= window:
        net_change = closes[-1] - closes[-window]
        total_path = float(np.sum(np.abs(np.diff(closes[-window:]))))
        trend_strength = net_change / (total_path + 1e-10)
        trend_strength = float(np.clip(trend_strength, -1.0, 1.0))
    else:
        trend_strength = 0.0

    # Hurst estimate
    hurst = estimate_hurst_simple(closes)

    # Volume ratio
    if volumes.size >= 20:
        recent_vol = float(np.mean(volumes[-10:]))
        avg_vol = float(np.mean(volumes[-60:])) if volumes.size >= 60 else float(np.mean(volumes))
        vol_ratio = recent_vol / (avg_vol + 1e-10)
        vol_ratio = float(np.clip(vol_ratio, 0.0, 5.0))
    else:
        vol_ratio = 1.0

    # Realized volatility (annualized)
    realized_vol = current_vol * np.sqrt(252) if current_vol > 0 else 0.0

    return np.array([
        float(vol_regime),
        float(trend_regime),
        float(liq_regime),
        vol_percentile,
        trend_strength,
        hurst,
        vol_ratio,
        realized_vol,
    ], dtype=np.float32)


def estimate_hurst_simple(closes: np.ndarray, max_lag: int = 20) -> float:
    """Simplified Hurst exponent estimate."""
    if closes.size < max_lag * 2:
        return 0.5

    lags = list(range(2, min(max_lag, closes.size // 2)))
    if len(lags) < 2:
        return 0.5

    tau = []
    for lag in lags:
        diffs = closes[lag:] - closes[:-lag]
        tau.append(np.std(diffs))

    tau = np.array(tau)
    lags = np.array(lags)

    if np.any(tau <= 0):
        return 0.5

    # Linear regression in log-log space
    log_lags = np.log(lags)
    log_tau = np.log(tau)

    # Simple least squares
    n = len(lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xy = np.sum(log_lags * log_tau)
    sum_xx = np.sum(log_lags ** 2)

    denom = n * sum_xx - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.5

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return float(np.clip(slope, 0.0, 1.0))


def detect_regime_change(regime_history: list[int], lookback: int = 10) -> bool:
    """
    Detect if there was a recent regime change.

    Args:
        regime_history: List of regime values (most recent last)
        lookback: Number of periods to check

    Returns:
        True if regime changed in the last `lookback` periods
    """
    if len(regime_history) < lookback:
        return False

    recent = regime_history[-lookback:]
    return len(set(recent)) > 1


REGIME_FEATURE_SIZE = 8  # Number of features returned by compute_regime_features

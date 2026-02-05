# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Extended technical indicators for feature engineering.

Provides 25+ indicators across momentum, volatility, volume, and trend categories.
All functions accept numpy arrays and return normalized float values.
"""

from __future__ import annotations

import numpy as np


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range - volatility indicator."""
    if closes.size < period + 1:
        return 0.0
    tr = np.zeros(closes.size)
    tr[0] = highs[0] - lows[0]
    for i in range(1, closes.size):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    return float(np.mean(tr[-period:]))


def atr_percent(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """ATR as percentage of price."""
    atr_val = atr(highs, lows, closes, period)
    if closes[-1] <= 0:
        return 0.0
    return atr_val / closes[-1] * 100.0


def bollinger_bands(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[float, float, float]:
    """Bollinger Bands: (upper, middle, lower)."""
    if closes.size < period:
        last = float(closes[-1]) if closes.size > 0 else 0.0
        return (last, last, last)
    window = closes[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window))
    upper = middle + num_std * std
    lower = middle - num_std * std
    return (upper, middle, lower)


def bollinger_pct_b(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    """Bollinger %B - position within bands (0-1 typically, can exceed)."""
    upper, middle, lower = bollinger_bands(closes, period, num_std)
    if upper == lower:
        return 0.5
    return (closes[-1] - lower) / (upper - lower)


def bollinger_bandwidth(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    """Bollinger Bandwidth - volatility measure."""
    upper, middle, lower = bollinger_bands(closes, period, num_std)
    if middle == 0:
        return 0.0
    return (upper - lower) / middle


def stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """Stochastic Oscillator: (%K, %D)."""
    if closes.size < k_period:
        return (50.0, 50.0)
    high_max = float(np.max(highs[-k_period:]))
    low_min = float(np.min(lows[-k_period:]))
    if high_max == low_min:
        k = 50.0
    else:
        k = (closes[-1] - low_min) / (high_max - low_min) * 100.0
    # %D is SMA of %K - approximate with current K
    d = k  # simplified; full version would track K history
    return (float(k), float(d))


def williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Williams %R - momentum oscillator (-100 to 0)."""
    if closes.size < period:
        return -50.0
    high_max = float(np.max(highs[-period:]))
    low_min = float(np.min(lows[-period:]))
    if high_max == low_min:
        return -50.0
    return (high_max - closes[-1]) / (high_max - low_min) * -100.0


def cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> float:
    """Commodity Channel Index."""
    if closes.size < period:
        return 0.0
    tp = (highs + lows + closes) / 3.0
    tp_window = tp[-period:]
    sma = float(np.mean(tp_window))
    mad = float(np.mean(np.abs(tp_window - sma)))
    if mad == 0:
        return 0.0
    return (tp[-1] - sma) / (0.015 * mad)


def roc(closes: np.ndarray, period: int = 10) -> float:
    """Rate of Change - momentum."""
    if closes.size < period + 1:
        return 0.0
    if closes[-period - 1] == 0:
        return 0.0
    return (closes[-1] - closes[-period - 1]) / closes[-period - 1] * 100.0


def momentum(closes: np.ndarray, period: int = 10) -> float:
    """Momentum - absolute price change."""
    if closes.size < period + 1:
        return 0.0
    return float(closes[-1] - closes[-period - 1])


def obv(closes: np.ndarray, volumes: np.ndarray) -> float:
    """On-Balance Volume - normalized."""
    if closes.size < 2 or volumes.size < 2:
        return 0.0
    obv_val = 0.0
    for i in range(1, closes.size):
        if closes[i] > closes[i - 1]:
            obv_val += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv_val -= volumes[i]
    # Normalize by total volume
    total_vol = float(np.sum(volumes))
    if total_vol == 0:
        return 0.0
    return obv_val / total_vol


def mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, period: int = 14) -> float:
    """Money Flow Index - volume-weighted RSI."""
    if closes.size < period + 1:
        return 50.0
    tp = (highs + lows + closes) / 3.0
    mf = tp * volumes
    pos_mf = 0.0
    neg_mf = 0.0
    for i in range(-period, 0):
        if tp[i] > tp[i - 1]:
            pos_mf += mf[i]
        elif tp[i] < tp[i - 1]:
            neg_mf += mf[i]
    if neg_mf == 0:
        return 100.0
    mf_ratio = pos_mf / neg_mf
    return 100.0 - (100.0 / (1.0 + mf_ratio))


def vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    """Volume Weighted Average Price."""
    if volumes.size == 0 or np.sum(volumes) == 0:
        return float(closes[-1]) if closes.size > 0 else 0.0
    tp = (highs + lows + closes) / 3.0
    return float(np.sum(tp * volumes) / np.sum(volumes))


def vwap_deviation(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    """Price deviation from VWAP as percentage."""
    vwap_val = vwap(highs, lows, closes, volumes)
    if vwap_val == 0:
        return 0.0
    return (closes[-1] - vwap_val) / vwap_val * 100.0


def keltner_channels(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20, mult: float = 2.0) -> tuple[float, float, float]:
    """Keltner Channels: (upper, middle, lower)."""
    if closes.size < period:
        last = float(closes[-1]) if closes.size > 0 else 0.0
        return (last, last, last)
    middle = float(np.mean(closes[-period:]))
    atr_val = atr(highs, lows, closes, period)
    upper = middle + mult * atr_val
    lower = middle - mult * atr_val
    return (upper, middle, lower)


def donchian_channels(highs: np.ndarray, lows: np.ndarray, period: int = 20) -> tuple[float, float, float]:
    """Donchian Channels: (upper, middle, lower)."""
    if highs.size < period:
        h = float(highs[-1]) if highs.size > 0 else 0.0
        l = float(lows[-1]) if lows.size > 0 else 0.0
        return (h, (h + l) / 2, l)
    upper = float(np.max(highs[-period:]))
    lower = float(np.min(lows[-period:]))
    middle = (upper + lower) / 2.0
    return (upper, middle, lower)


def historical_volatility(closes: np.ndarray, period: int = 20, annualize: bool = True) -> float:
    """Historical volatility of returns."""
    if closes.size < period + 1:
        return 0.0
    returns = np.diff(np.log(closes[-period - 1:]))
    vol = float(np.std(returns))
    if annualize:
        vol *= np.sqrt(252)  # annualize assuming daily data
    return vol


def garman_klass_volatility(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 20) -> float:
    """Garman-Klass volatility estimator - more efficient than close-to-close."""
    if closes.size < period:
        return 0.0
    n = period
    log_hl = np.log(highs[-n:] / lows[-n:]) ** 2
    log_co = np.log(closes[-n:] / opens[-n:]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return float(np.sqrt(np.mean(gk) * 252))


def hurst_exponent(closes: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent estimate - trend vs mean-reversion indicator.
    H > 0.5: trending
    H < 0.5: mean-reverting
    H = 0.5: random walk
    """
    if closes.size < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        pp = np.subtract(closes[lag:], closes[:-lag])
        tau.append(np.std(pp))
    tau = np.array(tau)
    lags = np.array(list(lags))
    # Linear fit in log-log space
    if np.any(tau <= 0) or np.any(lags <= 0):
        return 0.5
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(np.clip(poly[0], 0.0, 1.0))


def parabolic_sar_signal(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, af_start: float = 0.02, af_max: float = 0.2) -> float:
    """Parabolic SAR signal: 1 for bullish, -1 for bearish, 0 for neutral."""
    if closes.size < 5:
        return 0.0
    # Simplified SAR calculation
    trend = 1 if closes[-1] > closes[-5] else -1
    return float(trend)


def supertrend_signal(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 10, multiplier: float = 3.0) -> float:
    """Supertrend signal: 1 for bullish, -1 for bearish."""
    if closes.size < period + 1:
        return 0.0
    atr_val = atr(highs, lows, closes, period)
    hl2 = (highs[-1] + lows[-1]) / 2.0
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    if closes[-1] > upper_band:
        return -1.0  # Price above upper = bearish signal
    if closes[-1] < lower_band:
        return 1.0   # Price below lower = bullish signal
    return 0.0


def ichimoku_signals(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> tuple[float, float, float, float]:
    """
    Ichimoku components (normalized as signals):
    Returns: (tenkan_kijun_cross, price_vs_cloud, cloud_color, chikou_signal)
    All values in [-1, 1]
    """
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52

    if closes.size < senkou_b_period:
        return (0.0, 0.0, 0.0, 0.0)

    def midpoint(h, l, p):
        return (np.max(h[-p:]) + np.min(l[-p:])) / 2.0

    tenkan = midpoint(highs, lows, tenkan_period)
    kijun = midpoint(highs, lows, kijun_period)
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = midpoint(highs, lows, senkou_b_period)

    # Tenkan-Kijun cross signal
    tk_cross = 1.0 if tenkan > kijun else -1.0 if tenkan < kijun else 0.0

    # Price vs cloud
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    if closes[-1] > cloud_top:
        price_cloud = 1.0
    elif closes[-1] < cloud_bottom:
        price_cloud = -1.0
    else:
        price_cloud = 0.0

    # Cloud color (future trend)
    cloud_color = 1.0 if senkou_a > senkou_b else -1.0

    # Chikou (lagging) vs price 26 periods ago
    chikou = closes[-1]
    price_26_ago = closes[-26] if closes.size >= 26 else closes[0]
    chikou_signal = 1.0 if chikou > price_26_ago else -1.0

    return (tk_cross, price_cloud, cloud_color, chikou_signal)


def compute_all_indicators(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    periods: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute all available indicators and return as dict.

    Returns ~30 indicator values suitable for ML features.
    """
    if periods is None:
        periods = [14]  # default period

    period = periods[0] if periods else 14

    result = {}

    # Volatility indicators
    result["atr_pct"] = atr_percent(highs, lows, closes, period)
    result["bollinger_pct_b"] = bollinger_pct_b(closes, 20)
    result["bollinger_bandwidth"] = bollinger_bandwidth(closes, 20)
    result["hist_volatility"] = historical_volatility(closes, 20)
    result["hurst"] = hurst_exponent(closes, 20)

    # Momentum indicators
    stoch_k, stoch_d = stochastic(highs, lows, closes, period)
    result["stoch_k"] = stoch_k / 100.0  # normalize to 0-1
    result["stoch_d"] = stoch_d / 100.0
    result["williams_r"] = (williams_r(highs, lows, closes, period) + 100.0) / 100.0  # normalize to 0-1
    result["cci"] = np.clip(cci(highs, lows, closes, 20) / 200.0, -1.0, 1.0)  # normalize
    result["roc"] = np.clip(roc(closes, 10) / 10.0, -1.0, 1.0)  # normalize
    result["momentum"] = momentum(closes, 10)

    # Volume indicators
    result["obv"] = obv(closes, volumes)
    result["mfi"] = mfi(highs, lows, closes, volumes, period) / 100.0  # normalize to 0-1
    result["vwap_dev"] = np.clip(vwap_deviation(highs, lows, closes, volumes) / 5.0, -1.0, 1.0)

    # Trend indicators
    kelt_upper, kelt_mid, kelt_lower = keltner_channels(highs, lows, closes, 20)
    if kelt_upper != kelt_lower:
        result["keltner_pos"] = (closes[-1] - kelt_lower) / (kelt_upper - kelt_lower)
    else:
        result["keltner_pos"] = 0.5

    donc_upper, donc_mid, donc_lower = donchian_channels(highs, lows, 20)
    if donc_upper != donc_lower:
        result["donchian_pos"] = (closes[-1] - donc_lower) / (donc_upper - donc_lower)
    else:
        result["donchian_pos"] = 0.5

    result["sar_signal"] = parabolic_sar_signal(highs, lows, closes)
    result["supertrend"] = supertrend_signal(highs, lows, closes, 10, 3.0)

    # Ichimoku
    tk_cross, price_cloud, cloud_color, chikou = ichimoku_signals(highs, lows, closes)
    result["ichimoku_tk"] = tk_cross
    result["ichimoku_cloud"] = price_cloud
    result["ichimoku_color"] = cloud_color
    result["ichimoku_chikou"] = chikou

    return result


def indicator_feature_vector(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """
    Returns a numpy array of all indicator values in consistent order.
    """
    indicators = compute_all_indicators(opens, highs, lows, closes, volumes)
    # Fixed order for consistent feature vector
    keys = [
        "atr_pct", "bollinger_pct_b", "bollinger_bandwidth", "hist_volatility", "hurst",
        "stoch_k", "stoch_d", "williams_r", "cci", "roc", "momentum",
        "obv", "mfi", "vwap_dev",
        "keltner_pos", "donchian_pos", "sar_signal", "supertrend",
        "ichimoku_tk", "ichimoku_cloud", "ichimoku_color", "ichimoku_chikou",
    ]
    return np.array([indicators.get(k, 0.0) for k in keys], dtype=np.float32)


INDICATOR_FEATURE_SIZE = 22  # Number of features returned by indicator_feature_vector

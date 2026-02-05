# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""
Multi-timeframe feature computation.

Computes features across multiple timeframes (5m, 15m, 1h) and combines
them into a single feature vector for improved signal quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample_ohlcv(
    df: pd.DataFrame,
    target_tf: str,
    source_tf: str = "5m",
) -> pd.DataFrame:
    """
    Resample OHLCV data from source timeframe to target timeframe.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] and DatetimeIndex
        target_tf: Target timeframe (e.g., "15m", "1h", "4h")
        source_tf: Source timeframe (default "5m")

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            return df

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[col] = lower
    if col_map:
        df = df.rename(columns=col_map)

    # Resample
    resampled = df.resample(target_tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


def compute_mtf_features(
    df_5m: pd.DataFrame,
    window: int = 20,
) -> np.ndarray:
    """
    Compute multi-timeframe features from 5-minute data.

    Creates features for 5m, 15m, and 1h timeframes.

    Returns:
        Feature array with 15 elements:
        - 5m: return, volatility, rsi_approx, trend_strength, volume_ratio
        - 15m: return, volatility, rsi_approx, trend_strength, volume_ratio
        - 1h: return, volatility, rsi_approx, trend_strength, volume_ratio
    """
    features = []

    for tf in ["5m", "15min", "1h"]:
        if tf == "5m":
            df = df_5m
        else:
            df = resample_ohlcv(df_5m, tf)

        tf_features = _compute_tf_features(df, window)
        features.extend(tf_features)

    return np.array(features, dtype=np.float32)


def _compute_tf_features(df: pd.DataFrame, window: int) -> list[float]:
    """Compute features for a single timeframe."""
    if df.empty or len(df) < window:
        return [0.0, 0.0, 0.5, 0.0, 1.0]  # neutral defaults

    closes = df["close"].values
    volumes = df["volume"].values

    # Return over window
    if closes[0] != 0:
        ret = (closes[-1] - closes[-window]) / closes[-window] if len(closes) >= window else 0.0
    else:
        ret = 0.0
    ret = float(np.clip(ret, -0.5, 0.5))  # clip extreme values

    # Volatility (normalized)
    returns = np.diff(np.log(np.maximum(closes[-window:], 1e-10)))
    vol = float(np.std(returns)) if len(returns) > 1 else 0.0
    vol = float(np.clip(vol * 100, 0, 10))  # scale to reasonable range

    # Approximate RSI
    rsi = _approx_rsi(closes, min(14, len(closes) - 1))

    # Trend strength (efficiency ratio)
    if len(closes) >= window:
        net = abs(closes[-1] - closes[-window])
        path = float(np.sum(np.abs(np.diff(closes[-window:]))))
        trend = net / (path + 1e-10)
        trend = float(np.clip(trend, 0, 1))
    else:
        trend = 0.0

    # Volume ratio (recent vs average)
    if len(volumes) >= window:
        recent_vol = float(np.mean(volumes[-5:]))
        avg_vol = float(np.mean(volumes[-window:]))
        vol_ratio = recent_vol / (avg_vol + 1e-10)
        vol_ratio = float(np.clip(vol_ratio, 0, 5))
    else:
        vol_ratio = 1.0

    return [ret, vol, rsi, trend, vol_ratio]


def _approx_rsi(closes: np.ndarray, period: int) -> float:
    """Approximate RSI calculation."""
    if len(closes) < period + 1:
        return 0.5  # neutral

    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if avg_loss == 0:
        return 1.0 if avg_gain > 0 else 0.5

    rs = avg_gain / avg_loss
    rsi = 1.0 - (1.0 / (1.0 + rs))  # normalized to 0-1
    return float(rsi)


def build_mtf_observation(
    closes_5m: np.ndarray,
    volumes_5m: np.ndarray,
    timestamps: np.ndarray | None = None,
    window: int = 20,
) -> np.ndarray:
    """
    Build multi-timeframe observation from numpy arrays.

    This is a simpler version that works with numpy arrays directly
    when full OHLCV DataFrame is not available.

    Returns 9-element feature vector (3 per timeframe):
    - return, volatility, volume_ratio for each of 5m, 15m, 1h
    """
    features = []

    # 5m features (direct)
    features.extend(_compute_simple_features(closes_5m, volumes_5m, window))

    # 15m features (aggregate every 3 bars)
    closes_15m = _aggregate_closes(closes_5m, 3)
    volumes_15m = _aggregate_volumes(volumes_5m, 3)
    features.extend(_compute_simple_features(closes_15m, volumes_15m, window // 3 + 1))

    # 1h features (aggregate every 12 bars)
    closes_1h = _aggregate_closes(closes_5m, 12)
    volumes_1h = _aggregate_volumes(volumes_5m, 12)
    features.extend(_compute_simple_features(closes_1h, volumes_1h, max(window // 12, 2)))

    return np.array(features, dtype=np.float32)


def _aggregate_closes(closes: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate closes by taking every Nth value (simulating higher TF close)."""
    if factor <= 1 or closes.size < factor:
        return closes
    # Take every Nth close (last close of each period)
    indices = np.arange(factor - 1, closes.size, factor)
    return closes[indices]


def _aggregate_volumes(volumes: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate volumes by summing over periods."""
    if factor <= 1 or volumes.size < factor:
        return volumes
    n_periods = volumes.size // factor
    trimmed = volumes[:n_periods * factor].reshape(n_periods, factor)
    return trimmed.sum(axis=1)


def _compute_simple_features(closes: np.ndarray, volumes: np.ndarray, window: int) -> list[float]:
    """Compute simple features for a timeframe."""
    if closes.size < 2:
        return [0.0, 0.0, 1.0]

    # Return
    w = min(window, closes.size - 1)
    if closes[-w - 1] != 0:
        ret = (closes[-1] - closes[-w - 1]) / closes[-w - 1]
    else:
        ret = 0.0
    ret = float(np.clip(ret, -0.5, 0.5))

    # Volatility
    returns = np.diff(np.log(np.maximum(closes[-w:], 1e-10)))
    vol = float(np.std(returns) * 100) if returns.size > 1 else 0.0
    vol = float(np.clip(vol, 0, 10))

    # Volume ratio
    if volumes.size >= w:
        recent = float(np.mean(volumes[-max(w // 4, 1):]))
        avg = float(np.mean(volumes[-w:]))
        vol_ratio = recent / (avg + 1e-10)
        vol_ratio = float(np.clip(vol_ratio, 0, 5))
    else:
        vol_ratio = 1.0

    return [ret, vol, vol_ratio]


MTF_FEATURE_SIZE = 9  # 3 features per timeframe * 3 timeframes
MTF_FULL_FEATURE_SIZE = 15  # 5 features per timeframe * 3 timeframes (when using DataFrame)

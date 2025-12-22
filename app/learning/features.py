from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def _sma(values: np.ndarray, period: int) -> float:
    if values.size < period:
        return float(np.mean(values))
    return float(np.mean(values[-period:]))


def _ema(values: np.ndarray, period: int) -> float:
    if values.size < period:
        return float(np.mean(values))
    alpha = 2.0 / (period + 1.0)
    ema = values[-period]
    for val in values[-period + 1 :]:
        ema = alpha * val + (1.0 - alpha) * ema
    return float(ema)


def _rsi(values: np.ndarray, period: int) -> float:
    if values.size < period + 1:
        return 50.0
    deltas = np.diff(values[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains) if gains.size else 0.0
    avg_loss = np.mean(losses) if losses.size else 0.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def observation_size(window_size: int, feature_config: dict | None = None) -> int:
    base = window_size * 2 + 2
    if not feature_config:
        return base
    include_returns = feature_config.get("include_returns", True)
    sma_periods = feature_config.get("sma_periods", [])
    ema_periods = feature_config.get("ema_periods", [])
    rsi_periods = feature_config.get("rsi_periods", [])
    extra = 0
    if include_returns:
        extra += 1
    extra += len(sma_periods) + len(ema_periods) + len(rsi_periods)
    return base + extra


def build_observation(
    closes: Sequence[float],
    volumes: Sequence[float],
    window_size: int,
    position: float,
    cash_pct: float,
    feature_config: dict | None = None,
) -> np.ndarray:
    if not closes:
        raise ValueError("closes must contain at least one value")
    if len(volumes) != len(closes):
        raise ValueError("volumes and closes must be the same length")

    closes_arr = np.asarray(closes, dtype=np.float32)
    volumes_arr = np.asarray(volumes, dtype=np.float32)
    if closes_arr.size < window_size:
        pad_len = window_size - closes_arr.size
        closes_arr = np.pad(closes_arr, (pad_len, 0), mode="edge")
        volumes_arr = np.pad(volumes_arr, (pad_len, 0), mode="edge")

    closes_window = closes_arr[-window_size:]
    volumes_window = volumes_arr[-window_size:]

    last_close = closes_window[-1]
    if last_close == 0:
        last_close = 1.0
    norm_closes = closes_window / last_close - 1.0

    vol_mean = float(np.mean(volumes_window)) or 1.0
    norm_volumes = volumes_window / vol_mean - 1.0

    features = [
        norm_closes.astype(np.float32, copy=False),
        norm_volumes.astype(np.float32, copy=False),
        np.array([position, cash_pct], dtype=np.float32),
    ]

    if feature_config:
        if feature_config.get("include_returns", True):
            ret = (closes_window[-1] - closes_window[0]) / max(closes_window[0], 1e-6)
            features.append(np.array([ret], dtype=np.float32))
        for period in feature_config.get("sma_periods", []):
            features.append(np.array([_sma(closes_window, int(period))], dtype=np.float32))
        for period in feature_config.get("ema_periods", []):
            features.append(np.array([_ema(closes_window, int(period))], dtype=np.float32))
        for period in feature_config.get("rsi_periods", []):
            features.append(np.array([_rsi(closes_window, int(period))], dtype=np.float32))

    obs = np.concatenate(features)
    return obs

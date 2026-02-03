# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import logging

import numpy as np
import pandas as pd

from app.utils.gpu_state import is_gpu_disabled

INSAMPLES = 12
OUTSAMPLES = 12
N_FEATURES = 5
PRICE_COLS = slice(0, 4)
TARGET_COL = 3
VOLUME_COL = 4
EPS = 1e-6

logger = logging.getLogger(__name__)


@dataclass
class KerasReturnSignals:
    expected_return: float
    short_term_score: float
    up_prob: float
    downside_risk: float
    pred_log_returns: list[float]


def compute_keras_return_signals(
    frame: pd.DataFrame,
    model_path: str,
    interval: str = "5m",
    insamples: int = INSAMPLES,
    device: str = "auto",
) -> KerasReturnSignals | None:
    window = _window_from_frame(frame, interval=interval, insamples=insamples)
    if window is None:
        return None
    device = _resolve_device(device)
    model = _load_model(model_path, device)
    if model is None:
        return None
    pred_log_returns = _predict_log_returns(window, model)
    if pred_log_returns is None:
        return None
    expected_return = float(np.sum(pred_log_returns))
    weights = np.linspace(1.0, 0.2, num=len(pred_log_returns), dtype=np.float32)
    short_term_score = float(np.dot(pred_log_returns, weights) / weights.sum())
    up_prob = float(np.mean(pred_log_returns > 0.0))
    downside_risk = float(np.min(np.cumsum(pred_log_returns)))
    return KerasReturnSignals(
        expected_return=expected_return,
        short_term_score=short_term_score,
        up_prob=up_prob,
        downside_risk=downside_risk,
        pred_log_returns=[float(x) for x in pred_log_returns],
    )


def _window_from_frame(frame: pd.DataFrame, interval: str, insamples: int) -> np.ndarray | None:
    if frame is None or frame.empty:
        return None
    ohlcv = _resample_ohlcv(frame, interval=interval)
    if ohlcv is None or len(ohlcv) < insamples:
        return None
    window = ohlcv.tail(insamples)
    if len(window) != insamples:
        return None
    return window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)


def _resample_ohlcv(frame: pd.DataFrame, interval: str) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame = frame.copy()
            frame.index = pd.to_datetime(frame.index)
        except Exception:
            return None
    resampled = frame.sort_index()
    if interval != "5m":
        resampled = (
            resampled.resample("5min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
    return resampled


def _predict_log_returns(window: np.ndarray, model) -> np.ndarray | None:
    if window.shape[0] != INSAMPLES:
        return None
    x_feat, _ = _features_from_window(window)
    pred_log_return = model(np.expand_dims(x_feat, axis=0), training=False)
    pred_log_return = np.asarray(pred_log_return).squeeze(axis=0)
    if pred_log_return.shape[0] != OUTSAMPLES:
        return None
    return pred_log_return.astype(np.float32)


def _features_from_window(window: np.ndarray) -> tuple[np.ndarray, float]:
    window = window[:, :N_FEATURES]
    price = window[:, PRICE_COLS]
    price_feat = _log_returns(price)

    volume = window[:, VOLUME_COL:VOLUME_COL + 1]
    volume = np.log1p(volume)
    vol_mean = np.mean(volume, axis=0, keepdims=True)
    vol_std = np.std(volume, axis=0, keepdims=True)
    volume = np.divide(volume - vol_mean, vol_std + EPS)

    x_feat = np.concatenate([price_feat, volume], axis=1)
    last_close = float(window[-1, TARGET_COL])
    return x_feat.astype(np.float32), last_close


def _log_returns(values: np.ndarray) -> np.ndarray:
    prev = values[:-1]
    curr = values[1:]
    log_ret = np.log(curr + EPS) - np.log(prev + EPS)
    first = np.zeros_like(log_ret[:1])
    return np.concatenate([first, log_ret], axis=0)


@lru_cache(maxsize=4)
def _load_model(model_path: str, device: str):
    try:
        import tensorflow as tf  # noqa: F401
        import keras
    except Exception as exc:
        logger.warning("TensorFlow not available; keras_returns disabled (%s)", exc)
        return None
    _force_tf_cpu(tf, device)
    try:
        return keras.models.load_model(model_path)
    except Exception as exc:
        logger.warning("Failed to load Keras model at %s (%s)", model_path, exc)
        return None


def _force_tf_cpu(tf, device: str) -> None:
    if device != "cpu" and not is_gpu_disabled():
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception as exc:
        logger.debug("Failed to force TensorFlow CPU mode: %s", exc)


def _resolve_device(device: str) -> str:
    if is_gpu_disabled():
        return "cpu"
    if device == "auto":
        return "cuda"
    return device

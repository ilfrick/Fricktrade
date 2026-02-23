# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Return Ranker — sklearn GBM drop-in replacement for the Keras return scorer.

Implements the same public interface as compute_keras_return_signals so it can
be wired into the AI filter without any other code changes.

Features (FEATURE_NAMES) are derived purely from the OHLCV frame so they are
available at AI-filter time (before strategies run).  The same feature vector
is written by the collector script and consumed here, guaranteeing train/serve
consistency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature schema — must match collect_return_ranker_data.py exactly
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # Return statistics over last WINDOW bars
    "mean_ret",
    "std_ret",
    "momentum",
    "last_ret",
    # Volume
    "vol_z",
    # Intraday signal metrics (compute_signal_metrics_from_window)
    "signal_30m_return_pct",
    "signal_60m_return_pct",
    "signal_early_volume_pct",
    "signal_runup_pct",
    "signal_drawdown_pct",
    "signal_abs_move",
    "signal_runup_abs",
    "signal_drawdown_abs",
    # Trend / directional
    "adx",
    "plus_di",
    "minus_di",
    "trend_strength",
    # Momentum oscillators
    "rsi_14",
    "rsi_9",
    # Volatility / mean-reversion
    "bollinger_pct_b",
    "atr_pct",
    # Volume-weighted momentum
    "stoch_k",
    # News features (fetched by AI filter via fetch_news_features; default 0/lookback_hours when absent)
    "catalyst_flag",       # 1.0 if keyword/LLM catalyst matched, else 0.0
    "news_article_count",  # number of articles in lookback window (0 = no news)
    "news_recency_hours",  # hours since most recent article (capped at lookback; lower = fresher)
]

WINDOW = 20          # bars used for feature extraction
N_FEATURE = len(FEATURE_NAMES)
NEWS_LOOKBACK_HOURS = 12.0   # default recency cap when no news fetched


# ---------------------------------------------------------------------------
# Output dataclass — identical to KerasReturnSignals for seamless wiring
# ---------------------------------------------------------------------------

@dataclass
class KerasReturnSignals:
    """Matches app.signals.keras_returns.KerasReturnSignals exactly."""
    expected_return: float
    short_term_score: float
    up_prob: float
    downside_risk: float
    pred_log_returns: list[float]


# ---------------------------------------------------------------------------
# Public API — same signature as compute_keras_return_signals
# ---------------------------------------------------------------------------

def compute_return_ranker_signals(
    frame: pd.DataFrame,
    model_path: str,
    interval: str = "5m",
    insamples: int = WINDOW,
    device: str = "auto",
    catalyst: bool = False,
    news_features: dict | None = None,
) -> KerasReturnSignals | None:
    """Score a symbol's bar frame and return return-signal estimates.

    Drop-in replacement for compute_keras_return_signals.  `device` is
    accepted for interface compatibility but ignored (sklearn runs on CPU).

    Args:
        news_features: dict with keys {catalyst, article_count, recency_hours}
                       as returned by fetch_news_features().  When provided,
                       takes precedence over the `catalyst` bool argument.
    """
    features = extract_features(frame, window=insamples, interval=interval,
                                catalyst=catalyst, news_features=news_features)
    if features is None:
        return None
    model = _load_model(model_path)
    if model is None:
        return None
    try:
        x = features.reshape(1, -1)
        expected_return = float(model.predict(x)[0])
    except Exception as exc:
        logger.debug("return_ranker predict failed: %s", exc)
        return None

    if math.isnan(expected_return) or math.isinf(expected_return):
        return None

    # Derive secondary signals from the single regression output
    short_term_score = expected_return * 0.25          # 15m proxy
    up_prob = float(1.0 / (1.0 + math.exp(-expected_return / 0.005)))
    downside_risk = float(min(0.0, expected_return))

    # Synthesise a flat pred_log_returns list (12 steps) so callers that
    # iterate pred_log_returns don't break
    step = expected_return / 12.0
    pred_log_returns = [step] * 12

    return KerasReturnSignals(
        expected_return=expected_return,
        short_term_score=short_term_score,
        up_prob=up_prob,
        downside_risk=downside_risk,
        pred_log_returns=pred_log_returns,
    )


# ---------------------------------------------------------------------------
# Feature extraction (mirrors ai_filter._feature_vector / _latest_features)
# ---------------------------------------------------------------------------

def extract_features(
    frame: pd.DataFrame,
    window: int = WINDOW,
    interval: str = "5m",
    catalyst: bool = False,
    news_features: dict | None = None,
) -> np.ndarray | None:
    """Extract the FEATURE_NAMES vector from an OHLCV DataFrame.

    Returns None if there are insufficient bars.
    """
    if frame is None or frame.empty:
        return None
    close = _col(frame, "close")
    volume = _col(frame, "volume")
    high = _col(frame, "high")
    low = _col(frame, "low")
    if close is None or volume is None or len(close) < window + 1:
        return None

    returns = np.diff(close) / np.where(close[:-1] == 0, 1.0, close[:-1])
    if len(returns) < window:
        return None

    w_ret = returns[-window:]
    w_vol = volume[-window:] if len(volume) >= window else np.zeros(window)
    w_close = close[-(window + 1):]
    w_high = high[-(window + 1):] if high is not None and len(high) >= window + 1 else None
    w_low = low[-(window + 1):] if low is not None and len(low) >= window + 1 else None

    # --- base return stats ---
    mean_ret = float(np.mean(w_ret))
    std_ret = float(np.std(w_ret)) or 1e-6
    momentum = float(np.sum(w_ret))
    last_ret = float(w_ret[-1])

    # --- volume z-score ---
    vol_mean = float(np.mean(w_vol)) or 1e-6
    vol_std = float(np.std(w_vol)) or 1e-6
    vol_z = (float(w_vol[-1]) - vol_mean) / vol_std

    # --- intraday signal metrics ---
    from app.utils.signal_features import compute_signal_metrics_from_window
    sig = compute_signal_metrics_from_window(
        prices=w_close.tolist(),
        volumes=(volume[-window - 1:] if len(volume) >= window + 1 else w_vol).tolist(),
        interval=interval,
        highs=w_high.tolist() if w_high is not None else None,
        lows=w_low.tolist() if w_low is not None else None,
    )

    # --- ADX / trend ---
    from app.learning.features import _adx, _trend_strength
    if w_high is not None and w_low is not None:
        adx_v, plus_di, minus_di = _adx(w_high, w_low, w_close, 14)
        trend_str = _trend_strength(w_close, 14)
    else:
        adx_v, plus_di, minus_di, trend_str = 25.0, 50.0, 50.0, 0.0

    # --- extended indicators ---
    rsi_14 = _rsi(w_close, 14)
    rsi_9 = _rsi(w_close, 9)
    bb_pct_b = _bollinger_pct_b(w_close, 20)
    atr_p = _atr_pct(w_high, w_low, w_close, 14) if w_high is not None and w_low is not None else 0.0
    stoch_k = _stoch_k(w_high, w_low, w_close, 14) if w_high is not None and w_low is not None else 50.0

    # Resolve news features — news_features dict takes precedence over catalyst bool
    nf = news_features or {}
    catalyst_val = float(nf.get("catalyst", catalyst))
    article_count = float(nf.get("article_count", 0))
    recency_hours = float(nf.get("recency_hours", NEWS_LOOKBACK_HOURS))

    vec = np.array([
        mean_ret, std_ret, momentum, last_ret,
        vol_z,
        sig.get("signal_30m_return_pct", 0.0),
        sig.get("signal_60m_return_pct", 0.0),
        sig.get("signal_early_volume_pct", 0.0),
        sig.get("signal_runup_pct", 0.0),
        sig.get("signal_drawdown_pct", 0.0),
        sig.get("signal_abs_move", 0.0),
        sig.get("signal_runup_abs", 0.0),
        sig.get("signal_drawdown_abs", 0.0),
        adx_v, plus_di, minus_di, trend_str,
        rsi_14, rsi_9,
        bb_pct_b, atr_p, stoch_k,
        catalyst_val,
        article_count,
        recency_hours,
    ], dtype=np.float64)

    if np.any(~np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


# ---------------------------------------------------------------------------
# Model load / save helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        logger.debug("return_ranker model not found: %s", model_path)
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        logger.warning("return_ranker model load failed: %s", exc)
        return None


def model_needs_training(model_path: str, retrain_hours: int = 24) -> bool:
    """True if model file is absent or older than retrain_hours."""
    path = Path(model_path)
    if not path.exists():
        return True
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age > timedelta(hours=retrain_hours)


# ---------------------------------------------------------------------------
# Indicator helpers (stdlib + numpy only — no sklearn dependency here)
# ---------------------------------------------------------------------------

def _col(frame: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in frame.columns:
        return None
    return frame[name].astype(float).values


def _rsi(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains) if gains.size else 0.0
    avg_loss = np.mean(losses) if losses.size else 0.0
    if avg_loss == 0.0:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))


def _bollinger_pct_b(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    if len(closes) < period:
        return 0.5
    w = closes[-period:]
    mid = np.mean(w)
    std = np.std(w)
    upper = mid + num_std * std
    lower = mid - num_std * std
    if upper == lower:
        return 0.5
    return float((closes[-1] - lower) / (upper - lower))


def _atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1 or closes[-1] <= 0:
        return 0.0
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    return float(np.mean(tr[-period:]) / closes[-1] * 100.0)


def _stoch_k(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period:
        return 50.0
    high_max = float(np.max(highs[-period:]))
    low_min = float(np.min(lows[-period:]))
    if high_max == low_min:
        return 50.0
    return float((closes[-1] - low_min) / (high_max - low_min) * 100.0)

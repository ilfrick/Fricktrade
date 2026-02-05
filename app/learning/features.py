# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from app.utils.signal_features import compute_signal_metrics_from_window
from app.learning.indicators import indicator_feature_vector, INDICATOR_FEATURE_SIZE
from app.learning.regime import compute_regime_features, REGIME_FEATURE_SIZE
from app.learning.multi_timeframe import build_mtf_observation, MTF_FEATURE_SIZE

# Ordered list of risk config paths (dot-separated) and default values
_RISK_FEATURE_FIELDS = [
    ("max_daily_loss_pct", 0.0),
    ("max_position_size_pct", 0.0),
    ("max_portfolio_leverage", 1.0),
    ("max_short_exposure_pct", 0.0),
    ("max_positions", 0.0),
    ("cooldown_seconds", 0.0),
    ("hard_stop_pct", 0.0),
    ("trailing_stop_pct", 0.0),
    ("circuit_breaker_drawdown_pct", 0.0),
    ("vol_targeting.enabled", 0.0),
    ("vol_targeting.target_vol_pct", 0.0),
    ("vol_targeting.min_scale", 0.0),
    ("vol_targeting.max_scale", 0.0),
    ("stress.enabled", 0.0),
    ("stress.shock_pct", 0.0),
    ("liquidity_haircut.enabled", 0.0),
    ("liquidity_haircut.max_participation", 0.0),
    ("liquidity_haircut.min_session_volume", 0.0),
    ("liquidity_haircut.max_spread_pct", 0.0),
    ("liquidity_haircut.volume_haircut_pct", 0.0),
    ("var.enabled", 0.0),
    ("var.max_var_pct", 0.0),
    ("var.max_cvar_pct", 0.0),
    ("exposure_caps.enabled", 0.0),
    ("kill_switch_profiles.enabled", 0.0),
]

_ACCOUNT_FLAG_FIELDS = [
    "account_blocked",
    "trading_blocked",
    "trade_suspended_by_user",
]

_RISK_REASON_CODES = {
    "ok": 0.0,
    "risk_block": 1.0,
    "var_limit": 2.0,
    "cvar_limit": 2.5,
    "var_limit_broker": 2.7,
    "cvar_limit_broker": 2.8,
    "exposure_cap": 3.0,
    "limit_block": 4.0,
    "order_limit": 5.0,
    "cooldown": 6.0,
    "shorting_disabled": 7.0,
    "account_blocked": 8.0,
    "risk_disabled": 9.0,
}


def _sma(values: np.ndarray, period: int) -> float:
    """Calculate Simple Moving Average with period validation."""
    if period <= 0:
        return float(np.mean(values)) if values.size > 0 else 0.0
    if values.size < period:
        return float(np.mean(values)) if values.size > 0 else 0.0
    return float(np.mean(values[-period:]))


def _ema(values: np.ndarray, period: int) -> float:
    """Calculate Exponential Moving Average with period validation."""
    if period <= 0:
        return float(np.mean(values)) if values.size > 0 else 0.0
    if values.size < period:
        return float(np.mean(values)) if values.size > 0 else 0.0
    alpha = 2.0 / (period + 1.0)
    ema = values[-period]
    for val in values[-period + 1 :]:
        ema = alpha * val + (1.0 - alpha) * ema
    return float(ema)


def _rsi(values: np.ndarray, period: int) -> float:
    """Calculate Relative Strength Index with period validation."""
    if period <= 0:
        return 50.0
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


def _adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> tuple[float, float, float]:
    """
    Calculate ADX with +DI and -DI.

    Returns:
        (adx, plus_di, minus_di) - all in range 0-100
    """
    if period <= 0 or closes.size < period + 1:
        return (25.0, 50.0, 50.0)  # Neutral defaults

    n = closes.size
    # True Range
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Wilder's smoothing (EMA-style with alpha = 1/period)
    def wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        result = np.zeros_like(arr)
        # Initial sum for first smoothed value
        result[p - 1] = np.mean(arr[:p])
        alpha = 1.0 / p
        for i in range(p, len(arr)):
            result[i] = result[i - 1] * (1 - alpha) + arr[i] * alpha
        return result

    atr = wilder_smooth(tr, period)
    smooth_plus = wilder_smooth(plus_dm, period)
    smooth_minus = wilder_smooth(minus_dm, period)

    # +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period - 1, n):
        if atr[i] > 0:
            plus_di[i] = 100.0 * smooth_plus[i] / atr[i]
            minus_di[i] = 100.0 * smooth_minus[i] / atr[i]

    # DX and ADX
    dx = np.zeros(n)
    for i in range(period - 1, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom

    adx = wilder_smooth(dx, period)

    # Clamp values to valid range (can exceed 100 in edge cases due to smoothing)
    final_adx = float(np.clip(adx[-1], 0.0, 100.0))
    final_plus_di = float(np.clip(plus_di[-1], 0.0, 100.0))
    final_minus_di = float(np.clip(minus_di[-1], 0.0, 100.0))

    return (final_adx, final_plus_di, final_minus_di)


def _trend_strength(closes: np.ndarray, period: int) -> float:
    """
    Calculate trend strength as efficiency ratio.

    Returns value in [-1, 1]:
    - +1.0 = perfect uptrend
    - -1.0 = perfect downtrend
    - 0.0 = choppy/no trend
    """
    if period <= 0 or closes.size < period:
        return 0.0

    window = closes[-period:]
    net_change = window[-1] - window[0]
    total_path = np.sum(np.abs(np.diff(window)))

    if total_path < 1e-10:
        return 0.0

    return float(np.clip(net_change / total_path, -1.0, 1.0))


def observation_size(window_size: int, feature_config: dict | None = None) -> int:
    base = window_size * 2 + 3
    if not feature_config:
        return base
    include_returns = feature_config.get("include_returns", True)
    include_signal_features = feature_config.get("include_signal_features", False)
    include_risk_features = feature_config.get("include_risk_features", True)
    include_extended_indicators = feature_config.get("include_extended_indicators", False)
    include_regime_features = feature_config.get("include_regime_features", False)
    include_mtf_features = feature_config.get("include_mtf_features", False)
    sma_periods = feature_config.get("sma_periods", [])
    ema_periods = feature_config.get("ema_periods", [])
    rsi_periods = feature_config.get("rsi_periods", [])
    extra = 0
    if include_returns:
        extra += 1
    if include_signal_features:
        extra += 8
    if include_risk_features:
        extra += risk_feature_size(include_decision=True)
    if include_extended_indicators:
        extra += INDICATOR_FEATURE_SIZE
    if include_regime_features:
        extra += REGIME_FEATURE_SIZE
    if include_mtf_features:
        extra += MTF_FEATURE_SIZE
    extra += len(sma_periods) + len(ema_periods) + len(rsi_periods)
    adx_periods = feature_config.get("adx_periods", [])
    trend_strength_periods = feature_config.get("trend_strength_periods", [])
    extra += len(adx_periods) * 3  # ADX returns 3 values: adx, +DI, -DI
    extra += len(trend_strength_periods)
    return base + extra


def build_observation(
    closes: Sequence[float],
    volumes: Sequence[float],
    window_size: int,
    position: float,
    cash_pct: float,
    buying_power_pct: float,
    feature_config: dict | None = None,
    risk_features: list[float] | None = None,
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
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

    include_risk_features = feature_config.get("include_risk_features", True) if feature_config else True
    risk_vec = []
    if include_risk_features:
        risk_len = risk_feature_size(include_decision=True)
        if risk_features and len(risk_features) != risk_len:
            raise ValueError(f"risk_features length {len(risk_features)} != expected {risk_len}")
        risk_vec = risk_features or [0.0] * risk_len

    features = [
        norm_closes.astype(np.float32, copy=False),
        norm_volumes.astype(np.float32, copy=False),
        np.array([position, cash_pct, buying_power_pct], dtype=np.float32),
        np.array(risk_vec, dtype=np.float32) if risk_vec else np.array([], dtype=np.float32),
    ]

    if feature_config:
        if feature_config.get("include_returns", True):
            ret = (closes_window[-1] - closes_window[0]) / max(closes_window[0], 1e-6)
            features.append(np.array([ret], dtype=np.float32))
        if feature_config.get("include_signal_features", False):
            interval = str(feature_config.get("signal_interval") or feature_config.get("interval") or "5m")
            signal_vals = compute_signal_metrics_from_window(
                prices=closes_window.tolist(),
                volumes=volumes_window.tolist(),
                interval=interval,
            )
            features.append(
                np.array(
                    [
                        signal_vals.get("signal_30m_return_pct", 0.0),
                        signal_vals.get("signal_60m_return_pct", 0.0),
                        signal_vals.get("signal_early_volume_pct", 0.0),
                        signal_vals.get("signal_runup_pct", 0.0),
                        signal_vals.get("signal_drawdown_pct", 0.0),
                        signal_vals.get("signal_abs_move", 0.0),
                        signal_vals.get("signal_runup_abs", 0.0),
                        signal_vals.get("signal_drawdown_abs", 0.0),
                    ],
                    dtype=np.float32,
                )
            )
        for period in feature_config.get("sma_periods", []):
            features.append(np.array([_sma(closes_window, int(period))], dtype=np.float32))
        for period in feature_config.get("ema_periods", []):
            features.append(np.array([_ema(closes_window, int(period))], dtype=np.float32))
        for period in feature_config.get("rsi_periods", []):
            features.append(np.array([_rsi(closes_window, int(period))], dtype=np.float32))
        for period in feature_config.get("adx_periods", []):
            if highs is not None and lows is not None:
                highs_arr = np.asarray(highs, dtype=np.float32)
                lows_arr = np.asarray(lows, dtype=np.float32)
                if highs_arr.size < window_size:
                    highs_arr = np.pad(highs_arr, (window_size - highs_arr.size, 0), mode="edge")
                    lows_arr = np.pad(lows_arr, (window_size - lows_arr.size, 0), mode="edge")
                highs_window = highs_arr[-window_size:]
                lows_window = lows_arr[-window_size:]
                adx_val, plus_di, minus_di = _adx(highs_window, lows_window, closes_window, int(period))
                features.append(np.array([adx_val, plus_di, minus_di], dtype=np.float32))
            else:
                features.append(np.array([25.0, 50.0, 50.0], dtype=np.float32))
        for period in feature_config.get("trend_strength_periods", []):
            features.append(np.array([_trend_strength(closes_window, int(period))], dtype=np.float32))

        # Extended indicators (25+ technical indicators)
        if feature_config.get("include_extended_indicators", False):
            if highs is not None and lows is not None:
                highs_arr = np.asarray(highs, dtype=np.float32)
                lows_arr = np.asarray(lows, dtype=np.float32)
                if highs_arr.size < window_size:
                    highs_arr = np.pad(highs_arr, (window_size - highs_arr.size, 0), mode="edge")
                    lows_arr = np.pad(lows_arr, (window_size - lows_arr.size, 0), mode="edge")
                highs_window = highs_arr[-window_size:]
                lows_window = lows_arr[-window_size:]
                # Use closes as opens approximation if not available
                opens_window = np.roll(closes_window, 1)
                opens_window[0] = closes_window[0]
                indicator_vec = indicator_feature_vector(
                    opens_window, highs_window, lows_window, closes_window, volumes_window
                )
            else:
                # Approximate highs/lows from closes
                indicator_vec = indicator_feature_vector(
                    closes_window, closes_window, closes_window, closes_window, volumes_window
                )
            features.append(indicator_vec)

        # Regime features (volatility, trend, liquidity regimes)
        if feature_config.get("include_regime_features", False):
            regime_vec = compute_regime_features(closes_window, volumes_window)
            features.append(regime_vec)

        # Multi-timeframe features (5m, 15m, 1h)
        if feature_config.get("include_mtf_features", False):
            mtf_vec = build_mtf_observation(closes_arr, volumes_arr, window=window_size)
            features.append(mtf_vec)

    obs = np.concatenate(features)
    return obs


def risk_feature_size(include_decision: bool = True) -> int:
    size = len(_RISK_FEATURE_FIELDS) + len(_ACCOUNT_FLAG_FIELDS)
    if include_decision:
        size += 3  # allowed flag, action code, reason code
    return size


def risk_feature_vector(
    risk_cfg: dict | None,
    risk_outcome: dict | None = None,
    account_flags: dict | None = None,
    include_decision: bool = True,
) -> list[float]:
    """
    Build a stable, ordered vector of risk parameters plus the latest per-symbol risk decision.
    """
    risk_cfg = risk_cfg or {}

    def _get(path: str, default: float) -> float:
        parts = path.split(".")
        val: dict | float | int | bool = risk_cfg
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part, default)
            else:
                val = default
                break
        try:
            return float(val)
        except Exception:
            return float(default)

    values = []
    for path, default in _RISK_FEATURE_FIELDS:
        if path.endswith("enabled"):
            values.append(1.0 if bool(_get(path, default)) else 0.0)
        else:
            values.append(_get(path, default))
    flags = account_flags or {}
    for key in _ACCOUNT_FLAG_FIELDS:
        values.append(1.0 if bool(flags.get(key, False)) else 0.0)

    if not include_decision:
        return values

    outcome = risk_outcome or {}
    action = str(outcome.get("action", "hold")).lower()
    action_code = 0.0
    if action == "buy":
        action_code = 1.0
    elif action == "sell":
        action_code = -1.0
    allowed = 1.0 if bool(outcome.get("allowed", False)) else 0.0
    reason = str(outcome.get("reason", "ok"))
    reason_code = _RISK_REASON_CODES.get(reason, 0.0)
    values.extend([allowed, action_code, reason_code])
    return values

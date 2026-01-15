# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from app.utils.signal_features import compute_signal_metrics_from_window

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
}


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
    base = window_size * 2 + 3
    if not feature_config:
        return base
    include_returns = feature_config.get("include_returns", True)
    include_signal_features = feature_config.get("include_signal_features", False)
    include_risk_features = feature_config.get("include_risk_features", True)
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
    extra += len(sma_periods) + len(ema_periods) + len(rsi_periods)
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

    obs = np.concatenate(features)
    return obs


def risk_feature_size(include_decision: bool = True) -> int:
    size = len(_RISK_FEATURE_FIELDS)
    if include_decision:
        size += 3  # allowed flag, action code, reason code
    return size


def risk_feature_vector(
    risk_cfg: dict | None,
    risk_outcome: dict | None = None,
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

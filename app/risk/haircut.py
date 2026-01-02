# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations


def apply_haircuts(
    risk_cfg: dict,
    allowed_value: float,
    last_price: float,
    market_state: dict,
) -> tuple[float, list[str], dict[str, float]]:
    if allowed_value <= 0 or last_price <= 0:
        return 0.0, ["insufficient_value"], {}
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    stress_cfg = risk_cfg.get("stress", {}) or {}
    if stress_cfg.get("enabled", False):
        shock_pct = float(stress_cfg.get("shock_pct", 0.0))
        if shock_pct > 0:
            factor = max(0.0, 1.0 - shock_pct / 100.0)
            adjusted = allowed_value * factor
            if adjusted < allowed_value:
                reasons.append("stress_haircut")
                metrics["stress_haircut_pct"] = shock_pct
                allowed_value = adjusted
    liquidity_cfg = risk_cfg.get("liquidity_haircut", {}) or {}
    if liquidity_cfg.get("enabled", False):
        spread_pct = float(market_state.get("spread_pct", 0.0) or 0.0)
        max_spread = float(liquidity_cfg.get("max_spread_pct", 0.0) or 0.0)
        if max_spread > 0 and spread_pct > max_spread:
            reasons.append("illiquid_spread")
            metrics["spread_pct"] = spread_pct
            return 0.0, reasons, metrics
        session_volume = float(market_state.get("session_volume", 0.0) or 0.0)
        min_volume = float(liquidity_cfg.get("min_session_volume", 0.0) or 0.0)
        volume_haircut = float(liquidity_cfg.get("volume_haircut_pct", 0.0) or 0.0)
        if min_volume > 0 and session_volume < min_volume and volume_haircut > 0:
            factor = max(0.0, 1.0 - volume_haircut / 100.0)
            adjusted = allowed_value * factor
            if adjusted < allowed_value:
                reasons.append("illiquid_volume")
                metrics["liquidity_haircut_pct"] = volume_haircut
                metrics["session_volume"] = session_volume
                allowed_value = adjusted
        max_participation = float(liquidity_cfg.get("max_participation", 0.0) or 0.0)
        if max_participation > 0 and session_volume > 0:
            cap_value = session_volume * last_price * max_participation
            if cap_value < allowed_value:
                reasons.append("liquidity_haircut")
                metrics["max_participation"] = max_participation
                allowed_value = cap_value
    return max(allowed_value, 0.0), reasons, metrics

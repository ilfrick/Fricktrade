# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.risk.haircut import apply_haircuts


def test_stress_haircut_reduces_value():
    risk_cfg = {"stress": {"enabled": True, "shock_pct": 10.0}}
    value, reasons, metrics = apply_haircuts(risk_cfg, 1000.0, 10.0, {})
    assert value == 900.0
    assert "stress_haircut" in reasons
    assert metrics["stress_haircut_pct"] == 10.0


def test_liquidity_spread_blocks_trade():
    risk_cfg = {"liquidity_haircut": {"enabled": True, "max_spread_pct": 0.5}}
    value, reasons, metrics = apply_haircuts(risk_cfg, 1000.0, 10.0, {"spread_pct": 1.0})
    assert value == 0.0
    assert "illiquid_spread" in reasons
    assert metrics["spread_pct"] == 1.0

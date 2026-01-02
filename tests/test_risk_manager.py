# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.risk.manager import RiskManager


def test_can_open_trade_respects_limits():
    cfg = {
        "max_daily_loss_pct": 5.0,
        "max_position_size_pct": 10.0,
        "max_short_exposure_pct": 20.0,
        "max_portfolio_leverage": 2.0,
    }
    manager = RiskManager(cfg)
    assert manager.can_open_trade(5.0, 5.0, 1.0)
    assert not manager.can_open_trade(10.0, 5.0, 1.0)
    assert not manager.can_open_trade(5.0, 20.0, 1.0)
    assert not manager.can_open_trade(5.0, 5.0, 2.0)

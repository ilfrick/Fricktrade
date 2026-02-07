# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.risk.manager import RiskManager
from app.risk.config import RiskConfig


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


def test_can_open_trade_disabled():
    manager = RiskManager({"enabled": False})
    assert manager.can_open_trade(999.0, 999.0, 999.0)


def test_daily_loss_tracking():
    manager = RiskManager({"max_daily_loss_pct": 2.0})
    manager.record_pnl(-1.0)
    assert manager.daily_loss == -1.0
    assert manager.can_open_trade(0.0, 0.0, 0.0)
    manager.record_pnl(-1.5)
    # daily_loss is clamped to min(0, ...) so should be -2.5
    assert manager.daily_loss == -2.5
    assert not manager.can_open_trade(0.0, 0.0, 0.0)


def test_daily_loss_reset():
    manager = RiskManager({"max_daily_loss_pct": 2.0})
    manager.record_pnl(-3.0)
    assert not manager.can_open_trade(0.0, 0.0, 0.0)
    manager.reset_daily()
    assert manager.daily_loss == 0.0
    assert manager.can_open_trade(0.0, 0.0, 0.0)


def test_circuit_breaker():
    manager = RiskManager({"circuit_breaker_drawdown_pct": 5.0})
    assert not manager.should_circuit_break(4.9)
    assert manager.should_circuit_break(5.0)
    assert manager.should_circuit_break(10.0)


def test_circuit_breaker_disabled():
    manager = RiskManager({"enabled": False})
    assert not manager.should_circuit_break(100.0)


class TestCheckCooldown:
    def test_no_cooldown(self):
        manager = RiskManager({"cooldown_seconds": 0})
        now = datetime.now(timezone.utc)
        blocked, reason = manager.check_cooldown(now - timedelta(seconds=1), now)
        assert not blocked
        assert reason is None

    def test_no_last_trade(self):
        manager = RiskManager({"cooldown_seconds": 60})
        now = datetime.now(timezone.utc)
        blocked, reason = manager.check_cooldown(None, now)
        assert not blocked

    def test_cooldown_active(self):
        manager = RiskManager({"cooldown_seconds": 60})
        now = datetime.now(timezone.utc)
        blocked, reason = manager.check_cooldown(now - timedelta(seconds=30), now)
        assert blocked
        assert reason == "cooldown"

    def test_cooldown_expired(self):
        manager = RiskManager({"cooldown_seconds": 60})
        now = datetime.now(timezone.utc)
        blocked, reason = manager.check_cooldown(now - timedelta(seconds=61), now)
        assert not blocked


class TestCheckOrderLimits:
    def test_disabled(self):
        manager = RiskManager({})
        violates, reason = manager.check_order_limits(100, 50.0, {"enabled": False})
        assert not violates

    def test_max_qty(self):
        manager = RiskManager({})
        violates, _ = manager.check_order_limits(200, 10.0, {"enabled": True, "max_order_qty": 100})
        assert violates

    def test_max_notional(self):
        manager = RiskManager({})
        violates, _ = manager.check_order_limits(10, 100.0, {"enabled": True, "max_order_notional": 500})
        assert violates  # 10*100 = 1000 > 500

    def test_min_notional(self):
        manager = RiskManager({})
        violates, _ = manager.check_order_limits(1, 5.0, {"enabled": True, "min_order_notional": 10})
        assert violates  # 1*5 = 5 < 10

    def test_within_limits(self):
        manager = RiskManager({})
        violates, _ = manager.check_order_limits(
            10, 50.0,
            {"enabled": True, "max_order_qty": 100, "max_order_notional": 10000, "min_order_notional": 10},
        )
        assert not violates


class TestCheckExposureCaps:
    def test_disabled(self):
        violates, _ = RiskManager.check_exposure_caps(
            "AAPL", "buy", 10, 100.0, {}, {"enabled": False},
        )
        assert not violates

    def test_venue_cap_exceeded(self):
        portfolio = {
            "equity": 10000.0,
            "positions": {"GOOG": {"qty": 10, "value": 5000.0}},
        }
        venue_fn = lambda s: "US"
        violates, reason = RiskManager.check_exposure_caps(
            "AAPL", "buy", 100, 100.0, portfolio,
            {"enabled": True, "venues": {"US": 50.0}},
            venue_fn=venue_fn,
        )
        # Existing exposure 5000 + delta 10000 = 15000 / 10000 * 100 = 150% > 50%
        assert violates
        assert reason == "exposure_cap"

    def test_venue_cap_within(self):
        portfolio = {
            "equity": 100000.0,
            "positions": {},
        }
        venue_fn = lambda s: "US"
        violates, _ = RiskManager.check_exposure_caps(
            "AAPL", "buy", 1, 100.0, portfolio,
            {"enabled": True, "venues": {"US": 50.0}},
            venue_fn=venue_fn,
        )
        assert not violates

    def test_sector_cap_exceeded(self):
        portfolio = {
            "equity": 10000.0,
            "positions": {"GOOG": {"qty": 10, "value": 4000.0}},
        }
        sector_fn = lambda s: "tech"
        violates, _ = RiskManager.check_exposure_caps(
            "AAPL", "buy", 20, 100.0, portfolio,
            {"enabled": True, "sectors": {"tech": 30.0}},
            sector_fn=sector_fn,
        )
        # 4000 + 2000 = 6000 / 10000 * 100 = 60% > 30%
        assert violates

    def test_zero_equity(self):
        portfolio = {"equity": 0.0, "positions": {}}
        violates, _ = RiskManager.check_exposure_caps(
            "AAPL", "buy", 10, 100.0, portfolio,
            {"enabled": True, "venues": {"US": 50.0}},
            venue_fn=lambda s: "US",
        )
        assert not violates


class TestRiskConfig:
    def test_from_dict(self):
        cfg = RiskConfig.from_dict({
            "enabled": True,
            "max_daily_loss_pct": 2.5,
            "cooldown_seconds": 30,
            "unknown_key": "ignored",
        })
        assert cfg.enabled is True
        assert cfg.max_daily_loss_pct == 2.5
        assert cfg.cooldown_seconds == 30

    def test_defaults(self):
        cfg = RiskConfig()
        assert cfg.enabled is True
        assert cfg.max_daily_loss_pct == 0.0
        assert cfg.circuit_breaker_drawdown_pct == 100.0

    def test_roundtrip(self):
        cfg = RiskConfig(max_daily_loss_pct=3.0, cooldown_seconds=60)
        d = cfg.to_dict()
        cfg2 = RiskConfig.from_dict(d)
        assert cfg2.max_daily_loss_pct == 3.0
        assert cfg2.cooldown_seconds == 60


class TestDailyResetTimezone:
    def test_reset_uses_configured_timezone(self):
        """Daily loss resets at midnight ET, not midnight UTC."""
        et = ZoneInfo("US/Eastern")
        manager = RiskManager({"max_daily_loss_pct": 5.0}, tz=et)
        manager.record_pnl(-3.0)
        assert manager.daily_loss == -3.0

        # Simulate time just past midnight ET but still same day in UTC
        # e.g. 2025-06-15 00:05 ET = 2025-06-15 04:05 UTC
        fake_et_next_day = datetime(2025, 6, 15, 0, 5, tzinfo=et)
        with patch("app.risk.manager.datetime") as mock_dt:
            mock_dt.now.return_value = fake_et_next_day
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            manager.reset_daily()

        assert manager.daily_loss == 0.0

    def test_default_timezone_is_utc(self):
        manager = RiskManager({})
        assert manager._tz == timezone.utc

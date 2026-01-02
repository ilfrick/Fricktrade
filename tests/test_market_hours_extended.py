# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime
from zoneinfo import ZoneInfo

from app.utils.market import is_market_open, is_venue_extended


def _cfg(extended_enabled: bool) -> dict:
    return {
        "app": {"timezone": "America/New_York"},
        "market": {
            "open_mode": "any",
            "extended_hours": {"enabled": extended_enabled},
            "venues": [
                {
                    "name": "NYSE",
                    "timezone": "America/New_York",
                    "trading_hours": {
                        "open": "09:30",
                        "close": "16:00",
                        "extended_open": "04:00",
                        "extended_close": "20:00",
                    },
                    "holidays": [],
                }
            ],
        },
    }


def test_extended_hours_gate() -> None:
    tz = ZoneInfo("America/New_York")
    premarket = datetime(2026, 1, 7, 7, 0, tzinfo=tz)
    regular = datetime(2026, 1, 7, 10, 0, tzinfo=tz)

    cfg_disabled = _cfg(False)
    assert not is_market_open(cfg_disabled, now=premarket)
    assert is_market_open(cfg_disabled, now=regular)

    cfg_enabled = _cfg(True)
    assert is_market_open(cfg_enabled, now=premarket)
    assert is_venue_extended(cfg_enabled, "NYSE", now=premarket)
    assert is_market_open(cfg_enabled, now=regular)
    assert not is_venue_extended(cfg_enabled, "NYSE", now=regular)

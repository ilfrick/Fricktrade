# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime
from zoneinfo import ZoneInfo

from app.utils.market import is_market_open, next_market_open


def _multi_venue_cfg(trading_venues=None):
    """Config with BorsaItaliana + NYSE; optionally filtered by trading_venues."""
    market = {
        "open_mode": "any",
        "extended_hours": {"enabled": False},
        "venues": [
            {
                "name": "BorsaItaliana",
                "timezone": "Europe/Rome",
                "trading_hours": {"open": "09:00", "close": "17:30"},
                "holidays": [],
            },
            {
                "name": "NYSE",
                "timezone": "America/New_York",
                "trading_hours": {"open": "09:30", "close": "16:00"},
                "holidays": [],
            },
        ],
    }
    if trading_venues is not None:
        market["trading_venues"] = trading_venues
    return {"app": {"timezone": "UTC"}, "market": market}


def test_trading_venues_filters_to_nyse_only():
    """During BorsaItaliana-only hours, NYSE filter should return False."""
    # 10:00 Rome = 04:00 New York (NYSE closed, BorsaItaliana open)
    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime(2026, 1, 7, 10, 0, tzinfo=rome_tz)

    # Without filter: BorsaItaliana is open → True
    cfg_all = _multi_venue_cfg()
    assert is_market_open(cfg_all, now=now)

    # With filter: only NYSE counts → False
    cfg_nyse = _multi_venue_cfg(trading_venues=["NYSE"])
    assert not is_market_open(cfg_nyse, now=now)


def test_trading_venues_empty_falls_back_to_all():
    """Empty trading_venues list = backward compat (all venues checked)."""
    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime(2026, 1, 7, 10, 0, tzinfo=rome_tz)

    for tv in [[], None]:
        cfg = _multi_venue_cfg(trading_venues=tv)
        assert is_market_open(cfg, now=now)


def test_trading_venues_absent_falls_back_to_all():
    """No trading_venues key at all = backward compat."""
    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime(2026, 1, 7, 10, 0, tzinfo=rome_tz)

    cfg = _multi_venue_cfg()  # no trading_venues key
    assert is_market_open(cfg, now=now)


def test_next_market_open_respects_filter():
    """next_market_open should only consider filtered venues."""
    rome_tz = ZoneInfo("Europe/Rome")
    # 08:00 Rome on a Wednesday → both venues closed
    now = datetime(2026, 1, 7, 8, 0, tzinfo=rome_tz)

    # Without filter: BorsaItaliana opens at 09:00 Rome
    cfg_all = _multi_venue_cfg()
    nxt_all = next_market_open(cfg_all, now=now)
    assert nxt_all is not None
    rome = nxt_all.astimezone(rome_tz)
    assert rome.hour == 9 and rome.minute == 0

    # With NYSE filter: next open is 09:30 ET = 15:30 Rome
    cfg_nyse = _multi_venue_cfg(trading_venues=["NYSE"])
    nxt_nyse = next_market_open(cfg_nyse, now=now)
    assert nxt_nyse is not None
    rome_nyse = nxt_nyse.astimezone(rome_tz)
    assert rome_nyse.hour == 15 and rome_nyse.minute == 30


def test_trading_venues_nonexistent_venue_returns_false():
    """trading_venues with a name that matches no configured venue → False."""
    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime(2026, 1, 7, 10, 0, tzinfo=rome_tz)

    cfg = _multi_venue_cfg(trading_venues=["LSE"])
    assert not is_market_open(cfg, now=now)


def test_next_market_open_nonexistent_venue_returns_none():
    """next_market_open with no matching venues → None."""
    rome_tz = ZoneInfo("Europe/Rome")
    now = datetime(2026, 1, 7, 8, 0, tzinfo=rome_tz)

    cfg = _multi_venue_cfg(trading_venues=["LSE"])
    assert next_market_open(cfg, now=now) is None

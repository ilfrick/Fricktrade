# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


def is_market_open(cfg: dict, now: datetime | None = None) -> bool:
    app_cfg = cfg.get("app", {})
    market_cfg = cfg.get("market", {})
    mode = market_cfg.get("open_mode", "any")
    venues = _normalize_venues(market_cfg)
    if not venues:
        venues = [
            {
                "name": market_cfg.get("venue", "market"),
                "timezone": app_cfg.get("timezone", "UTC"),
                "trading_hours": market_cfg.get("trading_hours", {}),
                "holidays": market_cfg.get("holidays", []),
            }
        ]
    checks = [
        _is_venue_open(venue, now=now)
        for venue in venues
    ]
    return all(checks) if mode == "all" else any(checks)


def _normalize_venues(market_cfg: dict) -> list[dict]:
    venues = market_cfg.get("venues", [])
    if isinstance(venues, list):
        return [v for v in venues if isinstance(v, dict)]
    return []


def _is_venue_open(venue_cfg: dict, now: datetime | None = None) -> bool:
    tz = ZoneInfo(venue_cfg.get("timezone", "UTC"))
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    else:
        current = current.astimezone(tz)

    if current.weekday() >= 5:
        return False

    holidays = set(venue_cfg.get("holidays", []))
    if current.date().isoformat() in holidays:
        return False

    hours = venue_cfg.get("trading_hours", {})
    open_str = hours.get("open", "09:00")
    close_str = hours.get("close", "17:30")
    open_time = time.fromisoformat(open_str)
    close_time = time.fromisoformat(close_str)
    now_time = current.time()

    if open_time <= close_time:
        return open_time <= now_time <= close_time
    return now_time >= open_time or now_time <= close_time

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


def is_market_open(cfg: dict, now: datetime | None = None, allow_extended: bool | None = None) -> bool:
    app_cfg = cfg.get("app", {})
    market_cfg = cfg.get("market", {})
    mode = market_cfg.get("open_mode", "any")
    allow_extended = _resolve_extended_flag(market_cfg, allow_extended)
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
        _venue_session_state(venue, now=now, allow_extended=allow_extended)["open"]
        for venue in venues
    ]
    return all(checks) if mode == "all" else any(checks)


def is_venue_open(cfg: dict, venue_name: str, now: datetime | None = None, allow_extended: bool | None = None) -> bool:
    market_cfg = cfg.get("market", {})
    allow_extended = _resolve_extended_flag(market_cfg, allow_extended)
    venues = _normalize_venues(market_cfg)
    for venue in venues:
        if str(venue.get("name", "")).lower() == venue_name.lower():
            return _venue_session_state(venue, now=now, allow_extended=allow_extended)["open"]
    return False


def is_venue_extended(cfg: dict, venue_name: str, now: datetime | None = None) -> bool:
    market_cfg = cfg.get("market", {})
    allow_extended = _resolve_extended_flag(market_cfg, None)
    venues = _normalize_venues(market_cfg)
    for venue in venues:
        if str(venue.get("name", "")).lower() == venue_name.lower():
            return _venue_session_state(venue, now=now, allow_extended=allow_extended)["extended"]
    return False


def next_market_open(cfg: dict, now: datetime | None = None, allow_extended: bool | None = None) -> datetime | None:
    app_cfg = cfg.get("app", {})
    market_cfg = cfg.get("market", {})
    mode = market_cfg.get("open_mode", "any")
    allow_extended = _resolve_extended_flag(market_cfg, allow_extended)
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
    next_times = []
    for venue in venues:
        next_time = _next_venue_open(venue, now=now, allow_extended=allow_extended)
        if next_time is not None:
            next_times.append(next_time)
    if not next_times:
        return None
    if mode == "all":
        return max(next_times)
    return min(next_times)


def _normalize_venues(market_cfg: dict) -> list[dict]:
    venues = market_cfg.get("venues", [])
    if isinstance(venues, list):
        return [v for v in venues if isinstance(v, dict)]
    return []

def _resolve_extended_flag(market_cfg: dict, allow_extended: bool | None) -> bool:
    if allow_extended is not None:
        return bool(allow_extended)
    extended_cfg = market_cfg.get("extended_hours", {}) or {}
    return bool(extended_cfg.get("enabled", False))


def _venue_session_state(
    venue_cfg: dict,
    now: datetime | None = None,
    allow_extended: bool = False,
) -> dict:
    tz = ZoneInfo(venue_cfg.get("timezone", "UTC"))
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    else:
        current = current.astimezone(tz)

    if current.weekday() >= 5:
        return {"open": False, "regular": False, "extended": False}

    holidays = set(venue_cfg.get("holidays", []))
    if current.date().isoformat() in holidays:
        return {"open": False, "regular": False, "extended": False}

    hours = venue_cfg.get("trading_hours", {})
    open_str = hours.get("open", "09:00")
    close_str = hours.get("close", "17:30")
    open_time = time.fromisoformat(open_str)
    close_time = time.fromisoformat(close_str)
    now_time = current.time()

    regular_open = _time_in_window(open_time, close_time, now_time)
    extended_open = False
    if allow_extended:
        ext_open_str = hours.get("extended_open")
        ext_close_str = hours.get("extended_close")
        if ext_open_str and ext_close_str:
            ext_open_time = time.fromisoformat(ext_open_str)
            ext_close_time = time.fromisoformat(ext_close_str)
            extended_open = _time_in_window(ext_open_time, ext_close_time, now_time)
    return {
        "open": regular_open or extended_open,
        "regular": regular_open,
        "extended": extended_open and not regular_open,
    }


def _next_venue_open(
    venue_cfg: dict,
    now: datetime | None = None,
    allow_extended: bool = False,
) -> datetime | None:
    tz = ZoneInfo(venue_cfg.get("timezone", "UTC"))
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    else:
        current = current.astimezone(tz)

    if _venue_session_state(venue_cfg, now=current, allow_extended=allow_extended)["open"]:
        return current

    holidays = set(venue_cfg.get("holidays", []))
    hours = venue_cfg.get("trading_hours", {})
    open_str = hours.get("open", "09:00")
    close_str = hours.get("close", "17:30")
    if allow_extended:
        ext_open_str = hours.get("extended_open")
        ext_close_str = hours.get("extended_close")
        if ext_open_str and ext_close_str:
            open_str = ext_open_str
            close_str = ext_close_str
    open_time = time.fromisoformat(open_str)
    close_time = time.fromisoformat(close_str)

    for offset in range(0, 10):
        day = current.date().fromordinal(current.date().toordinal() + offset)
        if day.weekday() >= 5:
            continue
        if day.isoformat() in holidays:
            continue
        open_dt = datetime.combine(day, open_time, tzinfo=tz)
        if open_time <= close_time:
            if offset == 0:
                if current.time() < open_time:
                    return open_dt
                if current.time() <= close_time:
                    return current
                continue
            return open_dt
        if offset == 0:
            if current.time() < close_time:
                return current
            if current.time() >= open_time:
                return current
            return open_dt
        return open_dt
    return None


def _time_in_window(open_time: time, close_time: time, now_time: time) -> bool:
    if open_time <= close_time:
        return open_time <= now_time <= close_time
    return now_time >= open_time or now_time <= close_time

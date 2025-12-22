from __future__ import annotations

import argparse
import logging
import time
from datetime import date

import io
import json
import requests
import pandas as pd
import yaml


_VENUE_HANDLERS = {
    "nyse": "nyse",
    "nasdaq": "nasdaq",
    "borsaitaliana": "borsa",
    "borsa_italiana": "borsa",
    "borsa-italiana": "borsa",
}


def _normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace(".", "").replace("/", "_")


def _resolve_exchange(venue: dict) -> str | None:
    name = venue.get("name", "")
    key = _normalize_name(name)
    if key in _VENUE_HANDLERS:
        return _VENUE_HANDLERS[key]
    return None


def _fetch_tables(url: str) -> list[pd.DataFrame]:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text))


def _extract_dates(tables: list[pd.DataFrame], dayfirst: bool) -> set[date]:
    dates: set[date] = set()
    for table in tables:
        for column in table.columns:
            series = table[column]
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
            for value in parsed.dropna().dt.date:
                dates.add(value)
    return dates


def _parse_month_day(value: str, year: int) -> date | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if "," in text:
        text = text.split(",", 1)[1].strip()
    parsed = pd.to_datetime(f"{text} {year}", errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _compute_nyse_holidays(start: date, end: date) -> list[str]:
    tables = _fetch_tables("https://www.nyse.com/markets/hours-calendars")
    if not tables:
        return []
    holiday_table = None
    for table in tables:
        if "Holiday" in table.columns:
            holiday_table = table
            break
    if holiday_table is None:
        return []
    dates: set[date] = set()
    for column in holiday_table.columns:
        if str(column).isdigit():
            year = int(column)
            for value in holiday_table[column]:
                parsed = _parse_month_day(str(value), year)
                if parsed:
                    dates.add(parsed)
    filtered = sorted([d for d in dates if start <= d <= end])
    return [d.isoformat() for d in filtered]


def _compute_nasdaq_holidays(start: date, end: date) -> list[str]:
    tables = _fetch_tables("https://www.nasdaqtrader.com/Trader.aspx?id=Calendar")
    if not tables:
        return []
    table = tables[0]
    if table.shape[1] < 2:
        return []
    date_col = table.iloc[:, 0]
    status_col = table.iloc[:, -1].astype(str).str.lower()
    parsed = pd.to_datetime(date_col, errors="coerce")
    dates = set()
    for dt, status in zip(parsed, status_col, strict=False):
        if pd.isna(dt):
            continue
        if "closed" not in status:
            continue
        d = dt.date()
        if start <= d <= end:
            dates.add(d)
    return [d.isoformat() for d in sorted(dates)]


def _compute_borsa_holidays(start: date, end: date) -> list[str]:
    dates: set[date] = set()
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    for year in range(start.year, end.year + 1):
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/IT"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = json.loads(response.text)
        for entry in payload:
            d = pd.to_datetime(entry.get("date"), errors="coerce")
            if pd.isna(d):
                continue
            dates.add(d.date())
            if entry.get("localName", "").lower().startswith("pasqua"):
                good_friday = d.date() - pd.Timedelta(days=2)
                dates.add(good_friday)
    filtered = sorted([d for d in dates if start <= d <= end])
    return [d.isoformat() for d in filtered]


def update_holidays(config_path: str, years_ahead: int = 1) -> bool:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    market_cfg = cfg.setdefault("market", {})
    venues = market_cfg.get("venues", [])
    if not isinstance(venues, list):
        logging.warning("Market venues not configured; skipping holiday update.")
        return False

    start = date(date.today().year, 1, 1)
    end = date(date.today().year + years_ahead, 12, 31)
    updated = False
    for venue in venues:
        if not isinstance(venue, dict):
            continue
        handler = _resolve_exchange(venue)
        if not handler:
            logging.warning("Unknown venue for holiday update: %s", venue.get("name"))
            continue
        if handler == "nyse":
            holidays = _compute_nyse_holidays(start, end)
        elif handler == "nasdaq":
            holidays = _compute_nasdaq_holidays(start, end)
        elif handler == "borsa":
            holidays = _compute_borsa_holidays(start, end)
        else:
            holidays = []
        if venue.get("holidays") != holidays:
            venue["holidays"] = holidays
            updated = True
            logging.info("Updated holidays for %s (%s): %d entries", venue.get("name"), handler, len(holidays))

    if updated:
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
    return updated


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config/config.yaml")
    parser.add_argument("--interval-days", type=int, default=None)
    parser.add_argument("--years-ahead", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    while True:
        cfg = _load_config(args.config)
        update_cfg = cfg.get("market", {}).get("holiday_update", {})
        if not update_cfg.get("enabled", True):
            logging.info("Holiday updates disabled in config")
            break
        interval_days = args.interval_days if args.interval_days is not None else int(update_cfg.get("interval_days", 7))
        years_ahead = args.years_ahead if args.years_ahead is not None else int(update_cfg.get("years_ahead", 1))
        try:
            update_holidays(args.config, years_ahead=years_ahead)
        except Exception as exc:
            logging.exception("Holiday update failed: %s", exc)
        if args.once:
            break
        time.sleep(max(interval_days, 1) * 86400)


if __name__ == "__main__":
    main()

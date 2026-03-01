# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Earnings calendar fetcher via Alpha Vantage.

Fetches next earnings date per symbol and classifies a symbol's position
relative to its earnings window (pre / post / none).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_earnings_calendar(
    symbols: list[str],
    api_key: str,
    base_url: str = "https://www.alphavantage.co/query",
    timeout: int = 15,
) -> dict[str, Optional[str]]:
    """Return {symbol: 'YYYY-MM-DD'} next earnings date via Alpha Vantage.

    Uses the EARNINGS_CALENDAR function which returns a CSV file.
    Free tier: 25 requests/day.  We make one bulk request for all symbols.
    """
    if not api_key:
        logger.debug("ALPHA_VANTAGE_API_KEY not set — earnings calendar unavailable")
        return {s: None for s in symbols}

    result: dict[str, Optional[str]] = {s: None for s in symbols}
    symbol_set = set(symbols)

    try:
        params = urllib.parse.urlencode({
            "function": "EARNINGS_CALENDAR",
            "horizon": "3month",
            "apikey": api_key,
        })
        url = f"{base_url}?{params}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            sym = str(row.get("symbol", "") or "").upper()
            if sym not in symbol_set:
                continue
            # AV uses 'reportDate' or 'reportedDate'
            date_str = (row.get("reportDate") or row.get("reportedDate") or "").strip()
            if date_str:
                result[sym] = date_str

    except Exception as exc:
        logger.warning("fetch_earnings_calendar failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Window classification
# ---------------------------------------------------------------------------

def get_earnings_window(
    symbol: str,
    calendar: dict[str, Optional[str]],
    days_before: int = 2,
    days_after: int = 30,
) -> str:
    """Return 'pre', 'post', or 'none' for the symbol's earnings window.

    - 'pre'  : earnings date is within *days_before* calendar days from today
    - 'post' : earnings date was within *days_after* calendar days ago
    - 'none' : outside any window (or no data)
    """
    date_str = calendar.get(symbol)
    if not date_str:
        return "none"
    try:
        earnings_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return "none"
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    delta = (earnings_dt - now).days
    if -days_after <= delta <= days_before:
        return "pre" if delta >= 0 else "post"
    return "none"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cached_calendar(path: str) -> dict[str, Optional[str]]:
    """Load earnings calendar from JSON file. Returns empty dict on failure."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(k): (v if v else None) for k, v in data.items()}
    except Exception:
        return {}


def save_cached_calendar(path: str, data: dict[str, Optional[str]]) -> None:
    """Persist earnings calendar to JSON."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("save_cached_calendar failed: %s", exc)

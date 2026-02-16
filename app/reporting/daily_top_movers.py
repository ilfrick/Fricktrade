# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json
import html
import logging
import smtplib
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:  # pragma: no cover
    StockHistoricalDataClient = None
    StockBarsRequest = None
    StockSnapshotRequest = None
    TimeFrame = None

from app.data.scanner import load_symbol_venues, load_universe
from app.utils.market import is_venue_open
import argparse

from app.utils.config import load_config
from app.brokers.config_utils import get_alpaca_account_cfg


def _read_alertmanager_config(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _smtp_settings(cfg: dict) -> dict[str, str | list[str]]:
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {})
    email_cfg = report_cfg.get("email", {}) or {}
    if not email_cfg.get("use_alertmanager_config", True):
        return {
            "host": str(email_cfg.get("smtp_host", "")),
            "from": str(email_cfg.get("from", "")),
            "user": str(email_cfg.get("smtp_user", "")),
            "password": str(email_cfg.get("smtp_password", "")),
            "to": list(email_cfg.get("to", []) or []),
            "require_tls": bool(email_cfg.get("smtp_require_tls", True)),
            "hello": str(email_cfg.get("smtp_hello", "")),
        }
    alert_cfg = _read_alertmanager_config(
        Path(email_cfg.get("alertmanager_config_path", "/app/alertmanager/alertmanager.yml"))
    )
    global_cfg = alert_cfg.get("global", {}) if isinstance(alert_cfg, dict) else {}
    receivers = alert_cfg.get("receivers", []) if isinstance(alert_cfg, dict) else []
    to_list = []
    for receiver in receivers:
        if receiver.get("name") == "email-notifications":
            for item in receiver.get("email_configs", []) or []:
                if item.get("to"):
                    to_list.append(str(item.get("to")))
    require_tls = email_cfg.get("smtp_require_tls")
    if require_tls is None:
        require_tls = bool(global_cfg.get("smtp_require_tls", True))
    return {
        "host": str(global_cfg.get("smtp_smarthost", "")),
        "from": str(global_cfg.get("smtp_from", "")),
        "user": str(global_cfg.get("smtp_auth_username", "")),
        "password": str(global_cfg.get("smtp_auth_password", "")),
        "to": to_list,
        "require_tls": bool(require_tls),
        "hello": str(email_cfg.get("smtp_hello", "")) or str(global_cfg.get("smtp_hello", "")),
    }


def _send_email(subject: str, body: str, html_body: str | None, cfg: dict) -> None:
    settings = _smtp_settings(cfg)
    if not settings.get("host") or not settings.get("from") or not settings.get("to"):
        logging.warning("Daily report email skipped: SMTP config incomplete.")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["from"]
    msg["To"] = ", ".join(settings["to"])
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    host = settings["host"]
    if ":" in host:
        host_name, host_port = host.split(":", 1)
        port = int(host_port)
    else:
        host_name = host
        port = 587
    try:
        with smtplib.SMTP(host_name, port, timeout=30) as server:
            server.ehlo()
            if settings.get("hello"):
                server.helo(settings["hello"])
            if settings.get("require_tls", True):
                if server.has_extn("starttls"):
                    server.starttls()
                    server.ehlo()
                else:
                    logging.warning("Daily report SMTP server does not support STARTTLS.")
                    return
            if settings.get("user") and settings.get("password"):
                if server.has_extn("auth"):
                    server.login(settings["user"], settings["password"])
                else:
                    logging.warning("Daily report SMTP server does not support AUTH; sending without login.")
            server.send_message(msg)
    except Exception as exc:
        logging.warning("Daily report email failed: %s", exc)


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _active_brokers(cfg: dict) -> list[str]:
    brokers_cfg = cfg.get("brokers", {})
    active = []
    for name, data in brokers_cfg.items():
        if not isinstance(data, dict):
            continue
        if data.get("enabled", True):
            active.append(str(name))
    return active


def _broker_venues(cfg: dict, venues: dict[str, dict]) -> dict[str, list[str]]:
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
    mapping = report_cfg.get("broker_venues", {}) or {}
    result: dict[str, list[str]] = {}
    for broker in _active_brokers(cfg):
        broker_list = mapping.get(broker)
        if isinstance(broker_list, list) and broker_list:
            result[broker] = [str(v) for v in broker_list if v in venues]
        else:
            result[broker] = list(venues.keys())
    return result


def _venue_map(cfg: dict) -> dict[str, dict]:
    market_cfg = cfg.get("market", {})
    venues = market_cfg.get("venues", []) or []
    out = {}
    for venue in venues:
        if not isinstance(venue, dict):
            continue
        name = str(venue.get("name") or "").strip()
        if not name:
            continue
        out[name] = venue
    return out


def _venue_close_dt(venue_cfg: dict, now: datetime) -> datetime:
    tz_name = str(venue_cfg.get("timezone", "UTC"))
    tz = timezone.utc if tz_name.upper() == "UTC" else datetime.now().astimezone().tzinfo
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(tz_name)
    except Exception:
        pass
    hours = venue_cfg.get("trading_hours", {}) or {}
    close_str = hours.get("close", "17:30")
    close_time = dt_time.fromisoformat(close_str)
    local_day = now.astimezone(tz).date()
    return datetime.combine(local_day, close_time, tzinfo=tz)


def _alpaca_client(cfg: dict) -> StockHistoricalDataClient | None:
    if StockHistoricalDataClient is None:
        return None
    alpaca_cfg = get_alpaca_account_cfg(cfg)
    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    if not api_key or not api_secret:
        return None
    return StockHistoricalDataClient(api_key, api_secret)


def _resolve_universe(cfg: dict) -> list[str]:
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
    universe = report_cfg.get("universe")
    if universe is None:
        universe = cfg.get("data", {}).get("dynamic_symbols", {}).get("universe", "alpaca_active")
    max_universe = int(report_cfg.get("max_universe", 5000))
    alpaca_cfg = get_alpaca_account_cfg(cfg)
    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    return load_universe(api_key, api_secret, universe, max_universe=max_universe)


def _fetch_daily_bars(
    client: StockHistoricalDataClient,
    symbols: list[str],
    day: datetime,
    feed: str,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not symbols:
        return results
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    try:
        for chunk in _chunked(symbols, 200):
            request = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=feed,
            )
            bars = client.get_stock_bars(request)
            data = bars.data if hasattr(bars, "data") else {}
            if not isinstance(data, dict):
                continue
            for symbol, series in data.items():
                last = _last_bar(series)
                if last is None:
                    continue
                open_px = _bar_field(last, "open", ["o"])
                close_px = _bar_field(last, "close", ["c"])
                if open_px is None or close_px is None:
                    continue
                open_px = float(open_px)
                close_px = float(close_px)
                if open_px <= 0:
                    continue
                gain_pct = (close_px - open_px) / open_px * 100.0
                results[str(symbol)] = {
                    "open": open_px,
                    "close": close_px,
                    "gain_pct": gain_pct,
                }
    except Exception as exc:
        logging.warning("Daily top movers daily bars fetch failed: %s", exc)
    return results


def _fetch_intraday_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    day: datetime,
    feed: str,
) -> list[dict]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    try:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=feed,
        )
        bars = client.get_stock_bars(request)
        data = bars.data if hasattr(bars, "data") else {}
        if not isinstance(data, dict):
            return []
        series = data.get(symbol)
        if series is None:
            return []
        rows = []
        if hasattr(series, "iterrows"):
            for _, row in series.iterrows():
                rows.append(
                    {
                        "datetime": row.get("timestamp"),
                        "Open": row.get("open"),
                        "High": row.get("high"),
                        "Low": row.get("low"),
                        "Close": row.get("close"),
                        "Volume": row.get("volume"),
                    }
                )
            return rows
        if isinstance(series, list):
            for bar in series:
                rows.append(
                    {
                        "datetime": _bar_field(bar, "timestamp", ["t"]),
                        "Open": _bar_field(bar, "open", ["o"]),
                        "High": _bar_field(bar, "high", ["h"]),
                        "Low": _bar_field(bar, "low", ["l"]),
                        "Close": _bar_field(bar, "close", ["c"]),
                        "Volume": _bar_field(bar, "volume", ["v"]),
                    }
                )
            return rows
        return []
    except Exception as exc:
        logging.warning("Daily top movers intraday bars fetch failed for %s: %s", symbol, exc)
        return []


def _save_bars(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("datetime,Open,High,Low,Close,Volume\n")
        for row in rows:
            handle.write(
                f"{row['datetime']},{row['Open']},{row['High']},{row['Low']},{row['Close']},{row['Volume']}\n"
            )


def _status_path(output_dir: Path, venue: str, date_str: str) -> Path:
    return output_dir / "_status" / f"{venue}_{date_str}.json"


def _write_status(output_dir: Path, venue: str, date_str: str, state: str, note: str | None = None) -> None:
    payload = {
        "venue": venue,
        "date": date_str,
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if note:
        payload["note"] = note
    path = _status_path(output_dir, venue, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_report_body(output_dir: Path, venue: str, date_str: str, body: str) -> None:
    report_dir = output_dir / date_str / "_email"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"daily_report_{venue}.txt"
    report_path.write_text(body, encoding="utf-8")


def _save_report_html(output_dir: Path, venue: str, date_str: str, body: str) -> None:
    report_dir = output_dir / date_str / "_email"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"daily_report_{venue}.html"
    report_path.write_text(body, encoding="utf-8")


def _bar_field(obj: Any, name: str, aliases: list[str]) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    for alias in aliases:
        if hasattr(obj, alias):
            return getattr(obj, alias)
    if isinstance(obj, dict):
        if name in obj:
            return obj.get(name)
        for alias in aliases:
            if alias in obj:
                return obj.get(alias)
    return None


def _last_bar(series: Any) -> Any | None:
    if series is None:
        return None
    if hasattr(series, "empty"):
        if series.empty:
            return None
        return series.iloc[-1]
    if isinstance(series, list):
        return series[-1] if series else None
    return None


def _summarize_intraday(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    opens = [float(r["Open"]) for r in rows if r.get("Open") is not None]
    closes = [float(r["Close"]) for r in rows if r.get("Close") is not None]
    highs = [float(r["High"]) for r in rows if r.get("High") is not None]
    lows = [float(r["Low"]) for r in rows if r.get("Low") is not None]
    vols = [float(r["Volume"]) for r in rows if r.get("Volume") is not None]
    if not opens or not closes:
        return {}
    open_px = opens[0]
    close_px = closes[-1]
    total_vol = sum(vols) if vols else 0.0
    summary = {
        "open": open_px,
        "close": close_px,
        "high": max(highs) if highs else close_px,
        "low": min(lows) if lows else open_px,
        "total_vol": total_vol,
    }
    first_30 = min(30, len(closes))
    first_60 = min(60, len(closes))
    if first_30 >= 5:
        summary["first_30m_return_pct"] = (closes[first_30 - 1] - open_px) / open_px * 100.0
        summary["first_30m_vol_pct"] = (sum(vols[:first_30]) / total_vol * 100.0) if total_vol else 0.0
    if first_60 >= 5:
        summary["first_60m_return_pct"] = (closes[first_60 - 1] - open_px) / open_px * 100.0
    if highs and lows:
        summary["max_runup_pct"] = (summary["high"] - open_px) / open_px * 100.0
        summary["max_drawdown_pct"] = (summary["low"] - open_px) / open_px * 100.0
    return summary


def _signal_thresholds(cfg: dict) -> dict[str, float]:
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
    thresholds = report_cfg.get("signal_thresholds", {}) or {}
    return {
        "early_return_30m_pct": float(thresholds.get("early_return_30m_pct", 1.5)),
        "sustained_return_60m_pct": float(thresholds.get("sustained_return_60m_pct", 2.5)),
        "early_volume_pct": float(thresholds.get("early_volume_pct", 20.0)),
        "runup_pct": float(thresholds.get("runup_pct", 4.0)),
        "drawdown_pct": float(thresholds.get("drawdown_pct", -2.0)),
    }


def _signal_hints(summary: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    hints: list[str] = []
    if not summary:
        return hints
    if summary.get("first_30m_return_pct", 0.0) >= thresholds["early_return_30m_pct"]:
        hints.append(f"early momentum (30m +{thresholds['early_return_30m_pct']:.1f}% or more)")
    if summary.get("first_60m_return_pct", 0.0) >= thresholds["sustained_return_60m_pct"]:
        hints.append(f"sustained momentum (60m +{thresholds['sustained_return_60m_pct']:.1f}% or more)")
    if summary.get("first_30m_vol_pct", 0.0) >= thresholds["early_volume_pct"]:
        hints.append(f"early volume surge (>= {thresholds['early_volume_pct']:.0f}% in first 30m)")
    if summary.get("max_drawdown_pct", 0.0) <= thresholds["drawdown_pct"]:
        hints.append(f"volatile dip (intra-day drawdown <= {thresholds['drawdown_pct']:.1f}%)")
    if summary.get("max_runup_pct", 0.0) >= thresholds["runup_pct"]:
        hints.append(f"strong intraday run-up (>= {thresholds['runup_pct']:.1f}%)")
    return hints


def _first_move_time(rows: list[dict], threshold_pct: float) -> datetime | None:
    if not rows:
        return None
    open_px = rows[0].get("Open")
    if open_px is None:
        return None
    target = float(open_px) * (1.0 + threshold_pct / 100.0)
    for row in rows:
        close_px = row.get("Close")
        if close_px is not None and float(close_px) >= target:
            return row.get("datetime")
    return None


def _format_metrics(summary: dict[str, float]) -> str | None:
    if not summary:
        return None
    parts = []
    open_px = summary.get("open")
    if "first_30m_return_pct" in summary:
        parts.append(f"30m_return={summary['first_30m_return_pct']:.2f}%")
    if "first_60m_return_pct" in summary:
        parts.append(f"60m_return={summary['first_60m_return_pct']:.2f}%")
    if "first_30m_vol_pct" in summary:
        parts.append(f"early_vol={summary['first_30m_vol_pct']:.1f}%")
    if "max_runup_pct" in summary:
        parts.append(f"runup={summary['max_runup_pct']:.2f}%")
    if "max_drawdown_pct" in summary:
        parts.append(f"drawdown={summary['max_drawdown_pct']:.2f}%")
    if open_px is not None and "close" in summary:
        parts.append(f"abs_move={summary['close'] - open_px:.2f}")
    if open_px is not None and "high" in summary:
        parts.append(f"runup_abs={summary['high'] - open_px:.2f}")
    if open_px is not None and "low" in summary:
        parts.append(f"drawdown_abs={summary['low'] - open_px:.2f}")
    return ", ".join(parts) if parts else None


def _news_settings(cfg: dict) -> dict[str, Any]:
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
    return report_cfg.get("news", {}) or {}


def _decision_trace_settings(cfg: dict) -> dict[str, Any]:
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
    return report_cfg.get("decision_trace", {}) or {}


def _load_decision_traces(cfg: dict, date_str: str) -> dict[str, dict]:
    trace_cfg = _decision_trace_settings(cfg)
    if not trace_cfg.get("enabled", True):
        return {}
    output_dir = Path(trace_cfg.get("output_dir", "/data/reports/decision_trace"))
    dates = [date_str]
    try:
        day = datetime.fromisoformat(date_str).date()
        dates.append((day - timedelta(days=1)).isoformat())
    except Exception:
        pass
    latest: dict[str, dict] = {}
    for d in dates:
        path = output_dir / f"{d}.jsonl"
        if not path.exists():
            continue
        try:
            _load_traces_tail(path, latest)
        except Exception as exc:
            logging.warning("Decision trace load failed for %s: %s", path, exc)
    return latest


def _load_traces_tail(path: Path, latest: dict[str, dict], tail_bytes: int = 2 * 1024 * 1024) -> None:
    """Read only the tail of a trace file to find latest record per symbol."""
    file_size = path.stat().st_size
    if file_size == 0:
        return
    with open(path, "rb") as fh:
        offset = max(0, file_size - tail_bytes)
        fh.seek(offset)
        if offset > 0:
            fh.readline()  # discard partial first line
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            symbol = record.get("symbol")
            if not symbol:
                continue
            ts = record.get("ts")
            existing = latest.get(symbol)
            if not existing or (ts and existing.get("ts") and ts > existing["ts"]):
                latest[symbol] = record


def _alpaca_news_keys(cfg: dict) -> tuple[str, str, str]:
    news_cfg = cfg.get("news", {}) or {}
    alpaca_cfg = get_alpaca_account_cfg(cfg)
    api_key = str(news_cfg.get("api_key", "")) or str(alpaca_cfg.get("api_key", ""))
    api_secret = str(news_cfg.get("api_secret", "")) or str(alpaca_cfg.get("api_secret", ""))
    base_url = str(news_cfg.get("base_url", "https://data.alpaca.markets"))
    return api_key, api_secret, base_url


def _fetch_news_items(symbols: list[str], cfg: dict, day: datetime) -> dict[str, list[dict]]:
    report_news = _news_settings(cfg)
    if not report_news.get("enabled", True):
        return {}
    provider = str(report_news.get("provider", "alpaca")).lower()
    if provider != "alpaca":
        return {}
    api_key, api_secret, base_url = _alpaca_news_keys(cfg)
    if not api_key or not api_secret:
        return {}
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    max_headlines = int(report_news.get("max_headlines", 8))
    include_summary = bool(report_news.get("include_summaries", False))
    by_symbol: dict[str, list[dict]] = {s: [] for s in symbols}
    if not symbols:
        return by_symbol
    try:
        import requests

        chunk_size = 50
        for idx in range(0, len(symbols), chunk_size):
            chunk = symbols[idx : idx + chunk_size]
            page_token = None
            while True:
                params = {
                    "symbols": ",".join(chunk),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": 50,
                    "sort": "asc",
                }
                if page_token:
                    params["page_token"] = page_token
                resp = requests.get(
                    f"{base_url.rstrip('/')}/v1beta1/news",
                    params=params,
                    headers={
                        "APCA-API-KEY-ID": api_key,
                        "APCA-API-SECRET-KEY": api_secret,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                payload = resp.json()
                items = payload.get("news", payload if isinstance(payload, list) else [])
                for item in items:
                    created_at = _parse_time(item.get("created_at") or item.get("updated_at"))
                    headline = str(item.get("headline") or "")
                    summary = str(item.get("summary") or "")
                    for sym in item.get("symbols", []) or []:
                        if sym not in by_symbol:
                            continue
                        if len(by_symbol[sym]) >= max_headlines:
                            continue
                        entry = {"created_at": created_at, "headline": headline}
                        if include_summary and summary:
                            entry["summary"] = summary
                        by_symbol[sym].append(entry)
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
    except Exception as exc:
        logging.warning("Daily top movers news fetch failed: %s", exc)
    return by_symbol


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _query_prometheus(prom_url: str, query: str) -> dict:
    import requests

    resp = requests.get(f"{prom_url.rstrip('/')}/api/v1/query", params={"query": query}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("result", [])


def _query_prometheus_value(prom_url: str, query: str) -> float | None:
    try:
        result = _query_prometheus(prom_url, query)
    except Exception:
        return None
    if not result:
        return None
    try:
        return float(result[0].get("value", [0, 0])[1])
    except Exception:
        return None


def _diagnose_no_trade(prom_url: str, symbol: str) -> str:
    total_active = _query_prometheus_value(prom_url, "sum(count_over_time(symbol_active[24h]))")
    if total_active is None or total_active <= 0:
        return "metrics_unavailable"
    active = _query_prometheus_value(
        prom_url,
        f'sum(count_over_time(symbol_active{{symbol="{symbol}"}}[24h]))',
    )
    if active is None:
        return "not_in_active_universe"
    if active < 0.5:
        return "not_in_active_universe"
    open_orders = _query_prometheus_value(prom_url, f'sum(open_orders{{symbol="{symbol}"}})')
    if open_orders and open_orders > 0:
        return "open_order_pending"
    position_qty = _query_prometheus_value(prom_url, f'sum(position_qty{{symbol="{symbol}"}})')
    if position_qty and position_qty > 0:
        return "held_position_no_trade"
    return "no_signal_or_filtered"


def _build_skip_reasons(prom_url: str, symbol: str) -> list[str]:
    result = _query_prometheus(
        prom_url,
        f'increase(orders_skipped_total{{symbol="{symbol}"}}[1d])',
    )
    reasons = []
    for item in result:
        metric = item.get("metric", {})
        reason = metric.get("reason")
        value = float(item.get("value", [0, 0])[1] or 0.0)
        if reason and value > 0:
            reasons.append(f"{reason} ({int(value)})")
    return reasons


def _explain_with_ai(symbol: str, reasons: list[str], cfg: dict) -> str | None:
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
    explain_cfg = report_cfg.get("explain_ai", {}) or {}
    if not explain_cfg.get("enabled", False):
        return None
    provider = str(explain_cfg.get("provider", "ollama")).lower()
    if provider != "ollama":
        return None
    base_url = str(explain_cfg.get("base_url", "http://ollama:11434"))
    model = str(explain_cfg.get("model", "llama3.1:8b"))
    timeout_seconds = int(explain_cfg.get("timeout_seconds", 10))
    max_chars = int(explain_cfg.get("max_text_chars", 800))
    prompt = (
        "Explain briefly why the agent did not trade this symbol today. "
        f"Symbol: {symbol}. Reasons: {', '.join(reasons) if reasons else 'no signals or filters'}."
        " Keep it short and factual."
    )
    payload = {"model": model, "prompt": prompt[:max_chars], "stream": False}
    try:
        import requests

        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = str(data.get("response", "")).strip()
        return reply if reply else None
    except Exception as exc:
        logging.warning("AI explanation failed for %s: %s", symbol, exc)
        return None


def _render_report(
    broker: str,
    venue: str,
    date_str: str,
    movers: list[dict],
    cfg: dict,
) -> str:
    lines = [f"Daily Top Movers - {broker} - {venue} - {date_str}", ""]
    for idx, item in enumerate(movers, start=1):
        traded = "yes" if item.get("traded") else "no"
        reasons = item.get("reasons") or []
        lines.append(
            f"{idx}. {item['symbol']} gain={item['gain_pct']:.2f}% traded={traded}"
        )
        if reasons:
            lines.append(f"   reasons: {', '.join(reasons)}")
        metrics = item.get("metrics")
        if metrics:
            lines.append(f"   metrics: {metrics}")
        hints = item.get("hints") or []
        if hints:
            lines.append(f"   signals: {', '.join(hints)}")
        trace = item.get("decision_trace") or {}
        if trace:
            decision = trace.get("decision")
            stage = trace.get("stage")
            reason = trace.get("reason")
            action = trace.get("action")
            action_strategy = trace.get("action_strategy")
            broker = trace.get("broker")
            parts = []
            if decision:
                parts.append(f"decision={decision}")
            if stage:
                parts.append(f"stage={stage}")
            if reason:
                parts.append(f"reason={reason}")
            if action:
                parts.append(f"action={action}")
            if action_strategy:
                parts.append(f"strategy={action_strategy}")
            if broker:
                parts.append(f"broker={broker}")
            if parts:
                lines.append(f"   decision: {', '.join(parts)}")
            selected = trace.get("orchestrator_selected")
            if selected:
                lines.append(f"   orchestrator: {selected}")
            trace_signals = trace.get("signals") or []
            if trace_signals:
                brief = ", ".join(
                    f"{s.get('name')}={s.get('action')}" for s in trace_signals if s.get("name")
                )
                if brief:
                    lines.append(f"   signals_trace: {brief}")
        news_items = item.get("news") or []
        if news_items:
            before_move = item.get("news_before_move")
            flag = "yes" if before_move else "no"
            lines.append(f"   news: {len(news_items)} headlines; before_move={flag}")
            for entry in news_items:
                headline = entry.get("headline", "")
                created_at = entry.get("created_at")
                created_str = created_at.isoformat() if isinstance(created_at, datetime) else ""
                summary = entry.get("summary")
                lines.append(f"     - {created_str} {headline}")
                if summary:
                    lines.append(f"       summary: {summary}")
        if item.get("ai_reason"):
            lines.append(f"   ai: {item['ai_reason']}")
    return "\n".join(lines)


def _render_report_html(
    broker: str,
    venue: str,
    date_str: str,
    movers: list[dict],
) -> str:
    title = f"Daily Top Movers - {broker} - {venue} - {date_str}"
    rows = []
    for idx, item in enumerate(movers, start=1):
        traded = "yes" if item.get("traded") else "no"
        reasons = ", ".join(item.get("reasons") or [])
        metrics = item.get("metrics") or ""
        hints = ", ".join(item.get("hints") or [])
        trace = item.get("decision_trace") or {}
        trace_bits = []
        for key in ("decision", "stage", "reason", "action", "action_strategy", "broker"):
            value = trace.get(key)
            if value:
                trace_bits.append(f"{key}={value}")
        trace_line = ", ".join(trace_bits)
        orch = trace.get("orchestrator_selected")
        signals = trace.get("signals") or []
        signals_line = ", ".join(
            f"{s.get('name')}={s.get('action')}" for s in signals if s.get("name")
        )
        news_items = item.get("news") or []
        news_html = ""
        if news_items:
            news_html = "<ul>" + "".join(
                f"<li>{html.escape((n.get('created_at') or '').isoformat() if isinstance(n.get('created_at'), datetime) else '')} "
                f"{html.escape(n.get('headline') or '')}</li>"
                for n in news_items
            ) + "</ul>"
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(item['symbol'])}</td>"
            f"<td>{item['gain_pct']:.2f}%</td>"
            f"<td>{traded}</td>"
            f"<td>{html.escape(reasons)}</td>"
            f"<td>{html.escape(metrics)}</td>"
            f"<td>{html.escape(hints)}</td>"
            f"<td>{html.escape(trace_line)}</td>"
            f"<td>{html.escape(str(orch)) if orch else ''}</td>"
            f"<td>{html.escape(signals_line)}</td>"
            f"<td>{news_html}</td>"
            "</tr>"
        )
    return (
        "<html><body>"
        f"<h2>{html.escape(title)}</h2>"
        "<table border='1' cellspacing='0' cellpadding='6'>"
        "<thead><tr>"
        "<th>#</th><th>Symbol</th><th>Gain %</th><th>Traded</th><th>Reasons</th>"
        "<th>Metrics</th><th>Signals</th><th>Decision Trace</th>"
        "<th>Orchestrator</th><th>Signals Trace</th><th>News</th>"
        "</tr></thead><tbody>"
        + "".join(rows) +
        "</tbody></table>"
        "</body></html>"
    )


def run_daily_reports(config_path: str) -> None:
    cfg = load_config(config_path)
    report_cfg = cfg.get("reports", {}).get("daily_top_movers", {}) or {}
    if not report_cfg.get("enabled", False):
        logging.info("Daily top movers disabled")
        return
    client = _alpaca_client(cfg)
    if client is None:
        logging.warning("Daily top movers unavailable: Alpaca client not configured.")
        return
    prom_url = str(report_cfg.get("prometheus_url", "http://prometheus:9090"))
    output_dir = Path(report_cfg.get("output_dir", "/data/reports/daily_top_movers"))
    training_dir = Path(report_cfg.get("training_data_dir", cfg.get("backtest", {}).get("data_dir", "/data")))
    max_symbols = int(report_cfg.get("top_n", 10))
    feed = str(report_cfg.get("feed", cfg.get("data", {}).get("dynamic_symbols", {}).get("feed", "iex")))
    close_delay = int(report_cfg.get("close_delay_minutes", 5))
    thresholds = _signal_thresholds(cfg)
    feed_symbols = _resolve_universe(cfg)
    if not feed_symbols:
        logging.warning("Daily top movers: no symbols in universe")
        return
    symbol_venues = cfg.get("market", {}).get("symbol_venues", {}) or {}
    if report_cfg.get("use_symbol_venues_auto", False):
        auto_cfg = cfg.get("market", {}).get("symbol_venues_auto", {}) or {}
        exchange_map = auto_cfg.get("exchange_venue_map", {}) or {}
        max_symbols_map = int(auto_cfg.get("max_symbols", 50000))
        if exchange_map:
            alpaca_cfg = get_alpaca_account_cfg(cfg)
            try:
                symbol_venues = load_symbol_venues(
                    alpaca_cfg.get("api_key", ""),
                    alpaca_cfg.get("api_secret", ""),
                    max_symbols_map,
                    exchange_map,
                )
            except Exception as exc:
                logging.warning("Symbol venue auto-load failed: %s", exc)
    venues = _venue_map(cfg)
    brokers = _active_brokers(cfg)
    broker_venues = _broker_venues(cfg, venues)
    last_run: dict[str, str] = {}
    poll_seconds = int(report_cfg.get("check_interval_seconds", 60))
    logging.info("Daily top movers scheduler starting; brokers=%d venues=%d", len(brokers), len(venues))

    while True:
        now = datetime.now(timezone.utc)
        for venue_name, venue_cfg in venues.items():
            close_dt = _venue_close_dt(venue_cfg, now) + timedelta(minutes=close_delay)
            if now < close_dt:
                continue
            date_str = close_dt.date().isoformat()
            if last_run.get(venue_name) == date_str:
                continue
            if is_venue_open(cfg, venue_name, now=now):
                continue
            decision_traces = _load_decision_traces(cfg, date_str)
            _write_status(output_dir, venue_name, date_str, "running")
            daily_bars = _fetch_daily_bars(client, feed_symbols, close_dt, feed)
            if not daily_bars:
                logging.warning("Daily top movers: no daily bars for %s on %s", venue_name, date_str)
                _write_status(output_dir, venue_name, date_str, "done", note="no_daily_bars")
                last_run[venue_name] = date_str
                continue
            by_gain = sorted(
                daily_bars.items(),
                key=lambda item: item[1].get("gain_pct", 0.0),
                reverse=True,
            )
            sections = []
            sections_html = []
            for broker in brokers:
                if venue_name not in broker_venues.get(broker, []):
                    continue
                movers = []
                for symbol, data in by_gain:
                    if symbol_venues:
                        venue = symbol_venues.get(symbol)
                        if venue and venue != venue_name:
                            continue
                    movers.append(
                        {
                            "symbol": symbol,
                            "gain_pct": data.get("gain_pct", 0.0),
                        }
                    )
                    if len(movers) >= max_symbols:
                        break
                if not movers:
                    continue
                news_map = _fetch_news_items([m["symbol"] for m in movers], cfg, close_dt)
                for item in movers:
                    symbol = item["symbol"]
                    trace = decision_traces.get(symbol)
                    if trace:
                        item["decision_trace"] = trace
                    trades = _query_prometheus(
                        prom_url,
                        f'increase(trades_total{{symbol="{symbol}"}}[1d])',
                    )
                    traded = any(float(r.get("value", [0, 0])[1] or 0.0) > 0 for r in trades)
                    item["traded"] = traded
                    if not traded:
                        reasons = _build_skip_reasons(prom_url, symbol)
                        if not reasons:
                            reasons = [_diagnose_no_trade(prom_url, symbol)]
                        item["reasons"] = reasons
                        ai_reason = _explain_with_ai(symbol, reasons, cfg)
                        if ai_reason:
                            item["ai_reason"] = ai_reason
                    rows = _fetch_intraday_bars(client, symbol, close_dt, feed)
                    if rows:
                        summary = _summarize_intraday(rows)
                        metrics = _format_metrics(summary)
                        if metrics:
                            item["metrics"] = metrics
                        hints = _signal_hints(summary, thresholds)
                        if hints:
                            item["hints"] = hints
                        first_move = _first_move_time(rows, thresholds["early_return_30m_pct"])
                        open_time = rows[0].get("datetime")
                        day_dir = output_dir / date_str / broker / venue_name
                        _save_bars(day_dir / f"{symbol}_1m.csv", rows)
                        if report_cfg.get("training_enabled", True):
                            _save_bars(training_dir / f"{symbol}_{date_str}_1m.csv", rows)
                        news_items = news_map.get(symbol, [])
                        if news_items:
                            item["news"] = news_items
                            corr_window = int(_news_settings(cfg).get("correlation_window_minutes", 90))
                            if open_time and first_move:
                                cutoff = min(first_move, open_time + timedelta(minutes=corr_window))
                                item["news_before_move"] = any(
                                    isinstance(entry.get("created_at"), datetime)
                                    and entry["created_at"] <= cutoff
                                    for entry in news_items
                                )
                            elif open_time:
                                cutoff = open_time + timedelta(minutes=corr_window)
                                item["news_before_move"] = any(
                                    isinstance(entry.get("created_at"), datetime)
                                    and entry["created_at"] <= cutoff
                                    for entry in news_items
                                )
                sections.append(_render_report(broker, venue_name, date_str, movers, cfg))
                sections_html.append(_render_report_html(broker, venue_name, date_str, movers))
            if sections:
                subject = f"Daily Top Movers - {venue_name} - {date_str}"
                body = "\n\n".join(sections)
                html_body = "<hr/>".join(sections_html)
                _save_report_body(output_dir, venue_name, date_str, body)
                _save_report_html(output_dir, venue_name, date_str, html_body)
                _send_email(subject, body, html_body, cfg)
            _write_status(output_dir, venue_name, date_str, "done")
            last_run[venue_name] = date_str
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/app/config/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_daily_reports(args.config)


if __name__ == "__main__":
    main()

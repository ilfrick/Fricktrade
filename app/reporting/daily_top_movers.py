from __future__ import annotations

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
    return {
        "host": str(global_cfg.get("smtp_smarthost", "")),
        "from": str(global_cfg.get("smtp_from", "")),
        "user": str(global_cfg.get("smtp_auth_username", "")),
        "password": str(global_cfg.get("smtp_auth_password", "")),
        "to": to_list,
        "require_tls": bool(global_cfg.get("smtp_require_tls", True)),
        "hello": str(global_cfg.get("smtp_hello", "")),
    }


def _send_email(subject: str, body: str, cfg: dict) -> None:
    settings = _smtp_settings(cfg)
    if not settings.get("host") or not settings.get("from") or not settings.get("to"):
        logging.warning("Daily report email skipped: SMTP config incomplete.")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["from"]
    msg["To"] = ", ".join(settings["to"])
    msg.set_content(body)

    host = settings["host"]
    if ":" in host:
        host_name, host_port = host.split(":", 1)
        port = int(host_port)
    else:
        host_name = host
        port = 587
    with smtplib.SMTP(host_name, port, timeout=30) as server:
        if settings.get("hello"):
            server.helo(settings["hello"])
        if settings.get("require_tls", True):
            server.starttls()
        if settings.get("user") and settings.get("password"):
            server.login(settings["user"], settings["password"])
        server.send_message(msg)


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
    alpaca_cfg = cfg.get("brokers", {}).get("alpaca", {}) or {}
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
    alpaca_cfg = cfg.get("brokers", {}).get("alpaca", {}) or {}
    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    return load_universe(api_key, api_secret, universe, max_universe=max_universe)


def _fetch_daily_bars(client: StockHistoricalDataClient, symbols: list[str], day: datetime) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not symbols:
        return results
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    for chunk in _chunked(symbols, 200):
        request = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start, end=end)
        bars = client.get_stock_bars(request)
        for symbol, df in bars.data.items():
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            try:
                open_px = float(row["open"])
                close_px = float(row["close"])
            except Exception:
                continue
            if open_px <= 0:
                continue
            gain_pct = (close_px - open_px) / open_px * 100.0
            results[str(symbol)] = {
                "open": open_px,
                "close": close_px,
                "gain_pct": gain_pct,
            }
    return results


def _fetch_intraday_bars(client: StockHistoricalDataClient, symbol: str, day: datetime) -> list[dict]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    request = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute, start=start, end=end)
    bars = client.get_stock_bars(request)
    df = bars.data.get(symbol)
    if df is None or df.empty:
        return []
    rows = []
    for _, row in df.iterrows():
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


def _query_prometheus(prom_url: str, query: str) -> dict:
    import requests

    resp = requests.get(f"{prom_url.rstrip('/')}/api/v1/query", params={"query": query}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", {}).get("result", [])


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
        if item.get("ai_reason"):
            lines.append(f"   ai: {item['ai_reason']}")
    return "\n".join(lines)


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
    close_delay = int(report_cfg.get("close_delay_minutes", 5))
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
            alpaca_cfg = cfg.get("brokers", {}).get("alpaca", {}) or {}
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
            daily_bars = _fetch_daily_bars(client, feed_symbols, close_dt)
            if not daily_bars:
                continue
            by_gain = sorted(
                daily_bars.items(),
                key=lambda item: item[1].get("gain_pct", 0.0),
                reverse=True,
            )
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
                for item in movers:
                    symbol = item["symbol"]
                    trades = _query_prometheus(
                        prom_url,
                        f'increase(trades_total{{symbol="{symbol}"}}[1d])',
                    )
                    traded = any(float(r.get("value", [0, 0])[1] or 0.0) > 0 for r in trades)
                    item["traded"] = traded
                    if not traded:
                        reasons = _build_skip_reasons(prom_url, symbol)
                        if not reasons:
                            reasons = ["no_trades_detected"]
                        item["reasons"] = reasons
                        ai_reason = _explain_with_ai(symbol, reasons, cfg)
                        if ai_reason:
                            item["ai_reason"] = ai_reason
                    rows = _fetch_intraday_bars(client, symbol, close_dt)
                    if rows:
                        day_dir = output_dir / date_str / broker / venue_name
                        _save_bars(day_dir / f"{symbol}_1m.csv", rows)
                        if report_cfg.get("training_enabled", True):
                            _save_bars(training_dir / f"{symbol}_{date_str}_1m.csv", rows)
                subject = f"Daily Top Movers - {broker} - {venue_name} - {date_str}"
                body = _render_report(broker, venue_name, date_str, movers, cfg)
                _send_email(subject, body, cfg)
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

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Iterable

import requests

from app.brokers.config_utils import get_alpaca_account_cfg


def fetch_catalyst_symbols(
    symbols: Iterable[str],
    provider: str,
    base_url: str,
    api_key: str,
    api_secret: str,
    lookback_hours: int,
    keywords: list[str] | None = None,
    llm_cfg: dict | None = None,
    timeout_seconds: int = 10,
    retries: int = 2,
) -> dict[str, bool]:
    symbols = [s for s in symbols if s]
    if not symbols:
        return {}
    if provider != "alpaca":
        return {s: False for s in symbols}
    if not api_key or not api_secret:
        return {s: False for s in symbols}
    try:
        catalysts: dict[str, bool] = {}
        chunk_size = 50
        for idx in range(0, len(symbols), chunk_size):
            chunk = symbols[idx : idx + chunk_size]
            chunk_result = _fetch_alpaca_news(
                chunk,
                base_url=base_url,
                api_key=api_key,
                api_secret=api_secret,
                lookback_hours=lookback_hours,
                keywords=keywords or [],
                llm_cfg=llm_cfg,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
            catalysts.update(chunk_result)
        for symbol in symbols:
            catalysts.setdefault(symbol, False)
        return catalysts
    except Exception as exc:
        logging.warning("News fetch failed for %d symbols: %s", len(symbols), exc)
        return {s: False for s in symbols}


def fetch_catalyst_symbols_for_config(
    symbols: Iterable[str],
    news_cfg: dict,
    brokers_cfg: dict,
) -> dict[str, bool]:
    provider = str(news_cfg.get("provider", "alpaca"))
    alpaca_cfg = get_alpaca_account_cfg(brokers_cfg if isinstance(brokers_cfg, dict) else {})
    api_key = str(news_cfg.get("api_key", "")) or str(alpaca_cfg.get("api_key", ""))
    api_secret = str(news_cfg.get("api_secret", "")) or str(alpaca_cfg.get("api_secret", ""))
    if provider != "brokers":
        return fetch_catalyst_symbols(
            symbols,
            provider=provider,
            base_url=str(news_cfg.get("base_url", "https://data.alpaca.markets")),
            api_key=api_key,
            api_secret=api_secret,
            lookback_hours=int(news_cfg.get("lookback_hours", 12)),
            keywords=list(news_cfg.get("keywords", [])),
            llm_cfg=dict(news_cfg.get("llm", {}) or {}),
            timeout_seconds=int(news_cfg.get("timeout_seconds", 10)),
            retries=int(news_cfg.get("retries", 2)),
        )
    catalysts = {s: False for s in symbols}
    if alpaca_cfg.get("enabled", True):
        alpaca = fetch_catalyst_symbols(
            symbols,
            provider="alpaca",
            base_url=str(news_cfg.get("base_url", "https://data.alpaca.markets")),
            api_key=api_key,
            api_secret=api_secret,
            lookback_hours=int(news_cfg.get("lookback_hours", 12)),
            keywords=list(news_cfg.get("keywords", [])),
            llm_cfg=dict(news_cfg.get("llm", {}) or {}),
            timeout_seconds=int(news_cfg.get("timeout_seconds", 10)),
            retries=int(news_cfg.get("retries", 2)),
        )
        for sym, is_cat in alpaca.items():
            catalysts[sym] = catalysts.get(sym, False) or is_cat
    return catalysts


def _fetch_alpaca_news(
    symbols: list[str],
    base_url: str,
    api_key: str,
    api_secret: str,
    lookback_hours: int,
    keywords: list[str],
    llm_cfg: dict | None,
    timeout_seconds: int,
    retries: int,
) -> dict[str, bool]:
    url = f"{base_url.rstrip('/')}/v1beta1/news"
    params = {"symbols": ",".join(symbols), "limit": 50}
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    payload = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout_seconds)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException:
            if attempt >= retries:
                raise
            continue
    if payload is None:
        return {s: False for s in symbols}
    items = payload.get("news", payload if isinstance(payload, list) else [])

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    keywords_lower = [k.lower() for k in keywords]
    llm_cfg = llm_cfg or {}
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_max_items = int(llm_cfg.get("max_items", 25))
    llm_checked = 0

    catalysts: dict[str, bool] = {s: False for s in symbols}
    for item in items:
        created_at = _parse_time(item.get("created_at") or item.get("updated_at"))
        if created_at and created_at < cutoff:
            continue
        headline_text = (item.get("headline") or item.get("summary") or "")
        if llm_enabled:
            if llm_checked >= llm_max_items:
                continue
            llm_checked += 1
            if not _llm_catalyst_decision(headline_text, llm_cfg):
                continue
        else:
            headline = headline_text.lower()
            if keywords_lower and not any(k in headline for k in keywords_lower):
                continue
        for sym in item.get("symbols", []) or []:
            if sym in catalysts:
                catalysts[sym] = True
    return catalysts


def _llm_catalyst_decision(text: str, llm_cfg: dict) -> bool:
    if not text:
        return False
    provider = str(llm_cfg.get("provider", "ollama")).lower()
    if provider != "ollama":
        return False
    base_url = str(llm_cfg.get("base_url", "http://localhost:11434"))
    model = str(llm_cfg.get("model", "llama3.1:8b"))
    timeout_seconds = int(llm_cfg.get("timeout_seconds", 10))
    max_chars = int(llm_cfg.get("max_text_chars", 800))
    prompt = (
        "Classify if this headline indicates a tradeable catalyst within 24h. "
        "Respond only with JSON: {\"catalyst\": true|false}.\n"
        f"Headline: {text[:max_chars]}"
    )
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = str(data.get("response", ""))
    except requests.RequestException as exc:
        logging.warning("LLM catalyst request failed: %s", exc)
        return False
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False
    snippet = reply[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except Exception:
        return False
    return bool(parsed.get("catalyst", False))


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None

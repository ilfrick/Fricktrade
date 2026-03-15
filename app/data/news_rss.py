# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Multi-source news fetcher: broad RSS feeds, per-symbol RSS, and CryptoPanic API.

Provides articles from freely available sources to complement the primary
Alpaca news provider. All sources degrade gracefully — any fetch failure
returns an empty dict so trading is never blocked.

Symbol matching strategy:
  - Crypto: BTC/USD matched against {"btc", "bitcoin"} in headline+summary
  - Equities: ticker matched as a word token in headline+summary
  - CryptoPanic: explicit currency tags per post (most precise)
  - Yahoo Finance RSS: per-symbol URL (most precise for equities)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from xml.etree import ElementTree

import requests


# ---------------------------------------------------------------------------
# Symbol → keyword mapping for text-based matching
# ---------------------------------------------------------------------------

_CRYPTO_ALIASES: dict[str, list[str]] = {
    "BTC/USD":    ["bitcoin", "btc"],
    "ETH/USD":    ["ethereum", "eth", "ether"],
    "XRP/USD":    ["ripple", "xrp"],
    "ADA/USD":    ["cardano", "ada"],
    "SOL/USD":    ["solana", "sol"],
    "DOGE/USD":   ["dogecoin", "doge"],
    "MATIC/USD":  ["polygon", "matic"],
    "DOT/USD":    ["polkadot", "dot"],
    "AVAX/USD":   ["avalanche", "avax"],
    "LINK/USD":   ["chainlink", "link"],
    "UNI/USD":    ["uniswap", "uni"],
    "AAVE/USD":   ["aave"],
    "LTC/USD":    ["litecoin", "ltc"],
    "BCH/USD":    ["bitcoin cash", "bch"],
    "ATOM/USD":   ["cosmos", "atom"],
    "FIL/USD":    ["filecoin", "fil"],
    "HYPE/USD":   ["hyperliquid", "hype"],
    "PEPE/USD":   ["pepe"],
    "WIF/USD":    ["dogwifhat", "wif"],
    "TRUMP/USD":  ["trump coin", "trump meme"],
    "BONK/USD":   ["bonk"],
    "GRT/USD":    ["the graph", " grt "],
    "CRV/USD":    ["curve", " crv "],
    "RENDER/USD": ["render", "rndr"],
    "SUSHI/USD":  ["sushiswap", "sushi"],
    "XTZ/USD":    ["tezos", " xtz "],
    "BAT/USD":    ["basic attention", " bat "],
    "PAXG/USD":   ["paxgold", "paxg"],
    "ONDO/USD":   ["ondo finance", " ondo "],
    "SKY/USD":    ["sky protocol", " sky "],
    "POL/USD":    ["pol token", " pol "],
    "LPT/USD":    ["livepeer", " lpt "],
}


def _symbol_terms(symbol: str) -> list[str]:
    """Lowercase search terms for a symbol."""
    if "/" in symbol:
        base = symbol.split("/")[0].lower()
        extras = _CRYPTO_ALIASES.get(symbol, [])
        return list(dict.fromkeys([base] + extras))
    return [symbol.lower()]


def _article_matches(text: str, terms: list[str]) -> bool:
    tl = text.lower()
    for t in terms:
        if t in tl:
            return True
    return False


# ---------------------------------------------------------------------------
# Time parsing helpers
# ---------------------------------------------------------------------------

def _parse_rfc2822(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        v = value.rstrip("Z") + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Raw RSS/Atom fetcher (no external dependencies beyond requests)
# ---------------------------------------------------------------------------

_RSS_UA = "Fricktrade/3.0 news-aggregator"


def _fetch_rss(url: str, timeout: int = 10, retries: int = 1) -> list[dict]:
    """Fetch an RSS 2.0 or Atom feed and return raw article dicts."""
    headers = {"User-Agent": _RSS_UA}
    raw = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            raw = resp.text
            break
        except requests.RequestException as exc:
            if attempt >= retries:
                logging.debug("RSS fetch failed %s: %s", url, exc)
                return []

    if not raw:
        return []
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        logging.debug("RSS parse error %s: %s", url, exc)
        return []

    articles: list[dict] = []
    ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
    is_atom = root.tag.endswith("}feed") or root.tag == "feed"

    if is_atom:
        pfx = f"{{{ns}}}" if ns else ""
        for entry in root.findall(f"{pfx}entry"):
            def _txt(tag: str) -> str:
                el = entry.find(f"{pfx}{tag}")
                return (el.text or "").strip() if el is not None else ""
            pub_raw = _txt("published") or _txt("updated")
            articles.append({
                "headline": _txt("title"),
                "summary": _txt("summary") or _txt("content"),
                "source": url,
                "published_at": pub_raw,
                "_pub_dt": _parse_iso(pub_raw),
            })
    else:
        for item in root.findall(".//item"):
            def _t(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""
            pub_raw = _t("pubDate")
            src_el = item.find("source")
            articles.append({
                "headline": _t("title"),
                "summary": _t("description"),
                "source": (src_el.text or url).strip() if src_el is not None else url,
                "published_at": pub_raw,
                "_pub_dt": _parse_rfc2822(pub_raw),
            })

    return articles


def _clean(art: dict) -> dict:
    """Strip internal fields before storing."""
    return {k: v for k, v in art.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Broad (non-symbol-specific) RSS sources
# ---------------------------------------------------------------------------

_BROAD_FEEDS: dict[str, str] = {
    "coindesk":        "https://feeds.feedburner.com/CoinDesk",
    "cointelegraph":   "https://cointelegraph.com/rss",
    "bitcoin_magazine":"https://bitcoinmagazine.com/.rss/full/",
    "the_block":       "https://www.theblock.co/rss.xml",
    "reuters_business":"https://feeds.reuters.com/reuters/businessNews",
    "globe_newswire":  "https://www.globenewswire.com/RSSFeed/subjectcode/16-news-release/keyword/stocks",
    "pr_newswire":     "https://www.prnewswire.com/rss/news-releases-list.rss",
}


def _fetch_broad(
    symbols: list[str],
    enabled: list[str],
    cutoff: datetime,
    timeout: int,
    max_per_symbol: int,
    result: dict[str, list[dict]],
) -> None:
    terms_map = {s: _symbol_terms(s) for s in symbols}
    for name in enabled:
        url = _BROAD_FEEDS.get(name)
        if not url:
            continue
        try:
            articles = _fetch_rss(url, timeout=timeout)
        except Exception as exc:
            logging.debug("Broad RSS %s error: %s", name, exc)
            continue
        for art in articles:
            pub_dt = art.get("_pub_dt")
            if pub_dt and pub_dt < cutoff:
                continue
            text = f"{art['headline']} {art['summary']}"
            for sym, terms in terms_map.items():
                if len(result[sym]) < max_per_symbol and _article_matches(text, terms):
                    result[sym].append(_clean(art))


# ---------------------------------------------------------------------------
# Per-symbol Yahoo Finance RSS (equities)
# ---------------------------------------------------------------------------

def _fetch_yahoo_single(sym: str, cutoff: datetime, timeout: int, max_per_symbol: int) -> list[dict]:
    """Fetch Yahoo Finance RSS for a single equity symbol."""
    url = f"https://finance.yahoo.com/rss/headline?s={sym}"
    out: list[dict] = []
    try:
        articles = _fetch_rss(url, timeout=timeout)
    except Exception as exc:
        logging.debug("Yahoo Finance RSS %s error: %s", sym, exc)
        return out
    for art in articles:
        pub_dt = art.get("_pub_dt")
        if pub_dt and pub_dt < cutoff:
            continue
        if len(out) < max_per_symbol:
            out.append(_clean(art))
    return out


def _fetch_yahoo(
    symbols: list[str],
    cutoff: datetime,
    timeout: int,
    max_per_symbol: int,
    result: dict[str, list[dict]],
) -> None:
    equity_syms = [s for s in symbols if "/" not in s]
    if not equity_syms:
        return
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_fetch_yahoo_single, sym, cutoff, timeout, max_per_symbol): sym
                for sym in equity_syms}
        for fut in as_completed(futs, timeout=30):
            sym = futs[fut]
            try:
                result[sym].extend(fut.result()[:max_per_symbol - len(result[sym])])
            except Exception as exc:
                logging.debug("Yahoo Finance RSS %s future error: %s", sym, exc)


# ---------------------------------------------------------------------------
# Per-symbol Google News RSS
# ---------------------------------------------------------------------------

def _fetch_google_news_single(sym: str, cutoff: datetime, timeout: int, max_per_symbol: int) -> list[dict]:
    """Fetch Google News RSS for a single symbol."""
    query = sym.split("/")[0] if "/" in sym else sym
    url = (
        f"https://news.google.com/rss/search"
        f"?q={query}+%22{query}%22&hl=en&gl=US&ceid=US:en"
    )
    out: list[dict] = []
    try:
        articles = _fetch_rss(url, timeout=timeout)
    except Exception as exc:
        logging.debug("Google News RSS %s error: %s", sym, exc)
        return out
    for art in articles:
        pub_dt = art.get("_pub_dt")
        if pub_dt and pub_dt < cutoff:
            continue
        if len(out) < max_per_symbol:
            out.append(_clean(art))
    return out


def _fetch_google_news(
    symbols: list[str],
    cutoff: datetime,
    timeout: int,
    max_per_symbol: int,
    result: dict[str, list[dict]],
) -> None:
    if not symbols:
        return
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_fetch_google_news_single, sym, cutoff, timeout, max_per_symbol): sym
                for sym in symbols}
        for fut in as_completed(futs, timeout=30):
            sym = futs[fut]
            try:
                result[sym].extend(fut.result()[:max_per_symbol - len(result[sym])])
            except Exception as exc:
                logging.debug("Google News RSS %s future error: %s", sym, exc)


# ---------------------------------------------------------------------------
# CryptoPanic API (free tier — no key required for public posts)
# ---------------------------------------------------------------------------

def _fetch_cryptopanic(
    symbols: list[str],
    api_key: str,
    cutoff: datetime,
    timeout: int,
    max_per_symbol: int,
    result: dict[str, list[dict]],
) -> None:
    crypto_map: dict[str, str] = {}  # currency_code -> symbol
    for sym in symbols:
        if "/" in sym:
            crypto_map[sym.split("/")[0].upper()] = sym
    if not crypto_map:
        return

    params: dict = {"currencies": ",".join(crypto_map), "public": "true", "kind": "news"}
    if api_key:
        params["auth_token"] = api_key

    try:
        resp = requests.get(
            "https://cryptopanic.com/api/free/v1/posts/",
            params=params,
            headers={"User-Agent": _RSS_UA},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logging.warning("CryptoPanic fetch failed: %s", exc)
        return

    for post in data.get("results", []) or []:
        pub_dt = _parse_iso(str(post.get("created_at", "") or ""))
        if pub_dt and pub_dt < cutoff:
            continue
        title = str(post.get("title", "") or "")
        art = {
            "headline": title,
            "summary": title,
            "source": "cryptopanic",
            "published_at": str(post.get("created_at", "") or ""),
        }
        for currency in post.get("currencies", []) or []:
            code = str(currency.get("code", "") or "").upper()
            sym = crypto_map.get(code)
            if sym and len(result[sym]) < max_per_symbol:
                result[sym].append(art)


# ---------------------------------------------------------------------------
# Public API — mirrors fetch_raw_articles_for_config interface
# ---------------------------------------------------------------------------

def fetch_multi_source_articles(
    symbols: Iterable[str],
    sources_cfg: list[dict],
    lookback_hours: int,
    timeout_seconds: int = 10,
    max_per_symbol: int = 10,
) -> dict[str, list[dict]]:
    """Fetch articles from all configured additional sources.

    Returns {symbol: [article_dicts]} merged across enabled sources.
    All failures are caught and logged — never raises.
    """
    symbols = [s for s in symbols if s]
    if not symbols or not sources_cfg:
        return {s: [] for s in symbols}

    result: dict[str, list[dict]] = {s: [] for s in symbols}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    broad_enabled: list[str] = []
    yahoo_enabled = False
    google_enabled = False
    cryptopanic_cfg: dict | None = None

    for src in sources_cfg:
        if not isinstance(src, dict) or not src.get("enabled"):
            continue
        name = str(src.get("name", ""))
        if name in _BROAD_FEEDS:
            broad_enabled.append(name)
        elif name == "yahoo_finance_rss":
            yahoo_enabled = True
        elif name == "google_news_rss":
            google_enabled = True
        elif name == "cryptopanic":
            cryptopanic_cfg = src

    if broad_enabled:
        try:
            _fetch_broad(symbols, broad_enabled, cutoff, timeout_seconds, max_per_symbol, result)
        except Exception as exc:
            logging.warning("Broad RSS fetch error: %s", exc)

    if yahoo_enabled:
        try:
            _fetch_yahoo(symbols, cutoff, timeout_seconds, max_per_symbol, result)
        except Exception as exc:
            logging.warning("Yahoo Finance RSS error: %s", exc)

    if google_enabled:
        try:
            _fetch_google_news(symbols, cutoff, timeout_seconds, max_per_symbol, result)
        except Exception as exc:
            logging.warning("Google News RSS error: %s", exc)

    if cryptopanic_cfg is not None:
        try:
            _fetch_cryptopanic(
                symbols,
                api_key=str(cryptopanic_cfg.get("api_key", "") or ""),
                cutoff=cutoff,
                timeout=timeout_seconds,
                max_per_symbol=max_per_symbol,
                result=result,
            )
        except Exception as exc:
            logging.warning("CryptoPanic fetch error: %s", exc)

    return result


def fetch_multi_source_catalysts(
    symbols: Iterable[str],
    sources_cfg: list[dict],
    lookback_hours: int,
    keywords: list[str],
    timeout_seconds: int = 10,
) -> dict[str, bool]:
    """Derive catalyst flags from multi-source articles via keyword matching.

    Used alongside Ollama LLM classification — both paths run independently.
    """
    articles = fetch_multi_source_articles(
        symbols, sources_cfg, lookback_hours,
        timeout_seconds=timeout_seconds, max_per_symbol=5,
    )
    keywords_lower = [k.lower() for k in keywords]
    result: dict[str, bool] = {}
    for sym, arts in articles.items():
        if not keywords_lower:
            result[sym] = bool(arts)
        else:
            result[sym] = any(
                any(k in (a.get("headline", "") + " " + a.get("summary", "")).lower()
                    for k in keywords_lower)
                for a in arts
            )
    return result

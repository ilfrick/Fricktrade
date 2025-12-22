from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient


@dataclass
class ScanFilters:
    price_min: float
    price_max: float
    relative_volume_min: float
    premarket_gain_min_pct: float
    min_shares_traded: float
    max_spread_pct: float
    require_catalyst: bool
    strict_spread: bool


def scan_symbols(
    symbols: Iterable[str],
    api_key: str,
    api_secret: str,
    feed: str,
    filters: ScanFilters,
    catalyst_map: dict[str, bool] | None = None,
    max_symbols: int = 50,
    timeout_seconds: int = 10,
    retries: int = 2,
) -> list[str]:
    symbols = [s for s in symbols if s]
    if not symbols or not api_key or not api_secret:
        return []
    client = StockHistoricalDataClient(api_key, api_secret)
    results = []
    catalyst_map = catalyst_map or {}
    for chunk in _chunked(symbols, 100):
        snapshots = _fetch_snapshots(client, chunk, feed, timeout_seconds, retries)
        if snapshots is None:
            continue
        for symbol, snap in snapshots.items():
            price = getattr(getattr(snap, "latest_trade", None), "price", None) or getattr(
                getattr(snap, "minute_bar", None), "close", None
            )
            if not price:
                continue
            daily = getattr(snap, "daily_bar", None)
            prev = getattr(snap, "prev_daily_bar", None)
            volume = getattr(daily, "volume", 0.0) or 0.0
            prev_volume = getattr(prev, "volume", 0.0) or 0.0
            prev_close = getattr(prev, "close", 0.0) or 0.0
            rel_vol = (volume / prev_volume) if prev_volume else 1.0
            gain_pct = ((price - prev_close) / prev_close * 100.0) if prev_close else 0.0

            quote = getattr(snap, "latest_quote", None)
            spread_pct = None
            if quote and quote.ask_price and quote.bid_price and price:
                spread_pct = (quote.ask_price - quote.bid_price) / price * 100.0

            if not _passes_filters(filters, price, rel_vol, gain_pct, volume, spread_pct, catalyst_map.get(symbol, False)):
                continue
            results.append(symbol)
            if len(results) >= max_symbols:
                return results
    return results


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _fetch_snapshots(
    client: StockHistoricalDataClient,
    symbols: list[str],
    feed: str,
    timeout_seconds: int,
    retries: int,
):
    for attempt in range(retries + 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.get_stock_snapshot,
                StockSnapshotRequest(symbol_or_symbols=symbols, feed=_map_feed(feed)),
            )
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError:
                if attempt >= retries:
                    return None
            except Exception:
                if attempt >= retries:
                    return None
    return None


def load_universe(
    api_key: str,
    api_secret: str,
    universe: Iterable[str] | str,
    max_universe: int = 500,
) -> list[str]:
    if isinstance(universe, str) and universe == "alpaca_active":
        client = TradingClient(api_key, api_secret, raw_data=True)
        assets = client.get_all_assets()
        symbols = []
        for asset in assets:
            status = asset.get("status")
            tradable = asset.get("tradable")
            asset_class = asset.get("class") or asset.get("asset_class")
            if status != "active":
                continue
            if tradable is not True:
                continue
            if asset_class != "us_equity":
                continue
            symbol = asset.get("symbol")
            if not symbol:
                continue
            symbols.append(symbol)
            if len(symbols) >= max_universe:
                break
        return symbols
    if isinstance(universe, str):
        return [s.strip() for s in universe.split(",") if s.strip()]
    return [s for s in universe if s]


def _passes_filters(
    filters: ScanFilters,
    price: float,
    rel_vol: float,
    gain_pct: float,
    volume: float,
    spread_pct: float | None,
    has_catalyst: bool,
) -> bool:
    if not (filters.price_min <= price <= filters.price_max):
        return False
    if rel_vol < filters.relative_volume_min:
        return False
    if gain_pct < filters.premarket_gain_min_pct:
        return False
    if volume < filters.min_shares_traded:
        return False
    if filters.require_catalyst and not has_catalyst:
        return False
    if spread_pct is None and filters.strict_spread:
        return False
    if spread_pct is not None and spread_pct > filters.max_spread_pct:
        return False
    return True


def _map_feed(feed: str) -> DataFeed:
    if feed.lower() == "sip":
        return DataFeed.SIP
    return DataFeed.IEX

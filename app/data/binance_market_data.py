# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Binance OHLCV market-data provider for /USDT (and other non-/USD) crypto pairs.

Provides:
  - fetch_binance_bars()          — bulk kline fetch, returns {symbol: DataFrame}
  - BinanceMarketDataProvider     — drop-in provider for app/main.py loop
  - _HybridMarketDataProvider     — routes /USDT → Binance, others → primary provider
  - _build_binance_client_from_cfg() — builds a python-binance Client from full cfg dict

_market_state_from_df and _interval_seconds are inlined here (not imported from
app/main.py) to avoid circular imports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.utils.signal_features import compute_signal_metrics_from_window

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers inlined from app/main.py to avoid circular imports
# ---------------------------------------------------------------------------

def _interval_seconds(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    if interval.endswith("h"):
        return int(interval[:-1]) * 3600
    if interval.endswith("d"):
        return int(interval[:-1]) * 86400
    return 60


def _bars_for_lookback(lookback_days: int, interval: str) -> int:
    if interval.endswith("m"):
        minutes = max(int(interval[:-1]), 1)
        per_day = max(int(390 / minutes), 1)
    elif interval.endswith("h"):
        hours = max(int(interval[:-1]), 1)
        per_day = max(int(6.5 / hours), 1)
    elif interval.endswith("d"):
        per_day = 1
    else:
        per_day = 1
    return max(per_day * max(lookback_days, 1), 1)


def _session_gain_pct(data: pd.DataFrame, prices: list[float], mode: str) -> float:
    if data is None or data.empty or not prices:
        return 0.0
    try:
        if mode == "session":
            first_price = prices[0]
            return (prices[-1] - first_price) / first_price * 100.0 if first_price else 0.0
        if len(data.index) <= 1:
            return 0.0
        prev_close = data["Close"].iloc[-2]
        if prev_close:
            return (prices[-1] - prev_close) / prev_close * 100.0
        return 0.0
    except Exception:
        return 0.0


def _empty_market_state() -> dict:
    return {
        "prices": [],
        "volumes": [],
        "session_prices": [],
        "session_volumes": [],
        "qty": 1,
        "exposure_pct": 1.0,
        "short_exposure_pct": 0.0,
        "leverage": 1.0,
        "last_price": None,
        "opens": [],
        "highs": [],
        "lows": [],
        "session_opens": [],
        "session_highs": [],
        "session_lows": [],
        "session_bar_count": 0,
        "session_volume": 0.0,
        "relative_volume": 0.0,
        "session_gain_pct": 0.0,
        "spread_pct": None,
        "signal_30m_return_pct": 0.0,
    }


def _market_state_from_df(
    data: pd.DataFrame,
    lookback_days: int,
    interval: str,
    session_gain_mode: str,
) -> dict:
    if data is None or data.empty:
        return _empty_market_state()
    if isinstance(data.index, pd.MultiIndex):
        data = data.copy()
        data.index = data.index.get_level_values(-1)
    if "close" in data.columns:
        data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
    if not data.empty:
        data = data.copy()
        for col in ("Open", "High", "Low", "Close"):
            if col in data.columns:
                data[col] = data[col].ffill().bfill()
        if "Volume" in data.columns:
            data["Volume"] = data["Volume"].fillna(0.0)
    if "Close" not in data.columns:
        return _empty_market_state()

    close = data["Close"]
    volume = data["Volume"] if "Volume" in data else None
    open_ = data["Open"] if "Open" in data else None
    high = data["High"] if "High" in data else None
    low = data["Low"] if "Low" in data else None
    lookback_bars = _bars_for_lookback(lookback_days, interval)
    prices = close.iloc[-lookback_bars:].to_numpy(dtype=float).tolist()
    volumes = volume.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if volume is not None else []
    opens = open_.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if open_ is not None else []
    highs = high.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if high is not None else []
    lows = low.iloc[-lookback_bars:].to_numpy(dtype=float).tolist() if low is not None else []

    day_data = data
    idx = data.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
        day_mask = idx.date == idx[-1].date()
        if day_mask.any():
            day_data = data.loc[day_mask]

    session_close = day_data["Close"] if "Close" in day_data.columns else close
    session_volume_col = day_data["Volume"] if "Volume" in day_data.columns else volume
    session_open_col = day_data["Open"] if "Open" in day_data.columns else open_
    session_high_col = day_data["High"] if "High" in day_data.columns else high
    session_low_col = day_data["Low"] if "Low" in day_data.columns else low

    session_prices = session_close.to_numpy(dtype=float).tolist() if session_close is not None else []
    session_volumes = (
        session_volume_col.to_numpy(dtype=float).tolist()
        if session_volume_col is not None
        else []
    )
    session_opens = session_open_col.to_numpy(dtype=float).tolist() if session_open_col is not None else []
    session_highs = session_high_col.to_numpy(dtype=float).tolist() if session_high_col is not None else []
    session_lows = session_low_col.to_numpy(dtype=float).tolist() if session_low_col is not None else []

    last_price = prices[-1] if prices else None
    avg_volume = float(sum(volumes) / len(volumes)) if volumes else 0.0
    session_volume = float(sum(volumes)) if volumes else 0.0
    rel_volume = float(volumes[-1] / avg_volume) if avg_volume else 0.0
    session_gain_pct = _session_gain_pct(data, prices, session_gain_mode)
    last_bar_ts = data.index[-1].to_pydatetime()
    state = {
        "prices": prices,
        "volumes": volumes,
        "session_prices": session_prices,
        "session_volumes": session_volumes,
        "qty": 1,
        "exposure_pct": 1.0,
        "short_exposure_pct": 0.0,
        "leverage": 1.0,
        "last_price": last_price,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "session_opens": session_opens,
        "session_highs": session_highs,
        "session_lows": session_lows,
        "session_bar_count": len(session_prices),
        "session_volume": session_volume,
        "relative_volume": rel_volume,
        "session_gain_pct": session_gain_pct,
        "spread_pct": None,
        "last_bar_ts": last_bar_ts,
    }
    state.update(
        compute_signal_metrics_from_window(
            prices=prices,
            volumes=volumes,
            highs=highs or None,
            lows=lows or None,
            interval=interval,
        )
    )
    return state


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------

def _to_binance_symbol(symbol: str) -> str:
    """Convert 'DOT/USDT' or 'DOT/USD' → 'DOTUSDT'."""
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        if quote.upper() == "USD":
            quote = "USDT"
        return f"{base.upper()}{quote.upper()}"
    return symbol.upper()


# ---------------------------------------------------------------------------
# Binance interval mapping
# ---------------------------------------------------------------------------

def _binance_interval(interval: str) -> str:
    """Map Fricktrade interval strings ('5m', '1h', '1d') to Binance kline intervals."""
    if interval.endswith("m"):
        return interval          # '5m' → '5m'
    if interval.endswith("h"):
        return interval          # '1h' → '1h'
    if interval.endswith("d"):
        return interval          # '1d' → '1d'
    return "5m"


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_binance_bars(
    symbols: list[str],
    client: Any,
    interval: str,
    lookback_days: int,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV bars from Binance for a list of symbols.

    Args:
        symbols:      Fricktrade-format symbols, e.g. ['DOT/USDT', 'ADA/USDT'].
        client:       python-binance Client instance.
        interval:     Bar interval string, e.g. '5m', '1h'.
        lookback_days: Number of days of history to fetch.

    Returns:
        Dict mapping original symbol → DataFrame with lowercase columns
        (open/high/low/close/volume) and UTC DatetimeIndex.
    """
    if not symbols or client is None:
        return {}

    bn_interval = _binance_interval(interval)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        bn_sym = _to_binance_symbol(symbol)
        try:
            klines = client.get_klines(
                symbol=bn_sym,
                interval=bn_interval,
                startTime=start_ms,
                endTime=now_ms,
                limit=1000,
            )
            if not klines:
                log.debug("Binance bars: no data for %s", symbol)
                continue
            # kline format: [open_time, open, high, low, close, volume, close_time, ...]
            rows = []
            for k in klines:
                ts = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
                rows.append({
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            df = pd.DataFrame(rows)
            df.index = pd.DatetimeIndex(
                [pd.Timestamp(int(k[0]), unit="ms", tz="UTC") for k in klines],
                name="timestamp",
            )
            df = df.sort_index()
            result[symbol] = df
            log.debug("Binance bars: %d bars for %s", len(df), symbol)
        except Exception as exc:
            log.warning("Binance bars fetch failed for %s (%s): %s", symbol, bn_sym, exc)

    log.info("Binance bars: fetched %d/%d symbols", len(result), len(symbols))
    return result


# ---------------------------------------------------------------------------
# Market-data provider
# ---------------------------------------------------------------------------

class BinanceMarketDataProvider:
    """Market-data provider backed by Binance klines (for /USDT symbols).

    Follows the same interface as AlpacaMarketDataProvider:
      prepare(symbols) — bulk-fetch and cache
      __call__(symbol) — return cached state or empty
    """

    def __init__(
        self,
        client: Any,
        lookback_days: int,
        interval: str,
        session_gain_mode: str = "gap",
    ) -> None:
        self._client = client
        self._lookback_days = lookback_days
        self._interval = interval
        self._session_gain_mode = session_gain_mode
        self._cache: dict[str, dict] = {}
        self._cache_ts: float = 0.0

    def prepare(self, symbols: list[str]) -> None:
        """Bulk-fetch bars for all /USDT (non-/USD) symbols and cache results."""
        usdt_symbols = [s for s in symbols if "/" in s and not s.upper().endswith("/USD")]
        if not usdt_symbols:
            return
        cache_ttl = _interval_seconds(self._interval)
        now = datetime.now(timezone.utc).timestamp()
        if now - self._cache_ts < cache_ttl:
            return  # Cache still fresh

        bars = fetch_binance_bars(
            usdt_symbols,
            self._client,
            self._interval,
            self._lookback_days,
        )
        self._cache = {}
        for symbol, df in bars.items():
            state = _market_state_from_df(
                df, self._lookback_days, self._interval, self._session_gain_mode
            )
            self._cache[symbol] = state
        self._cache_ts = now

    def __call__(self, symbol: str) -> dict:
        if symbol in self._cache:
            return self._cache[symbol]
        # Individual fetch fallback
        bars = fetch_binance_bars(
            [symbol], self._client, self._interval, self._lookback_days
        )
        if symbol in bars:
            state = _market_state_from_df(
                bars[symbol], self._lookback_days, self._interval, self._session_gain_mode
            )
            self._cache[symbol] = state
            return state
        return _empty_market_state()


# ---------------------------------------------------------------------------
# Hybrid provider
# ---------------------------------------------------------------------------

class _HybridMarketDataProvider:
    """Routes /USDT symbols to BinanceMarketDataProvider, all others to primary."""

    def __init__(self, primary: Any, binance_provider: BinanceMarketDataProvider) -> None:
        self._primary = primary
        self._binance = binance_provider

    def prepare(self, symbols: list[str]) -> None:
        non_usdt = [s for s in symbols if not ("/" in s and not s.upper().endswith("/USD"))]
        usdt = [s for s in symbols if "/" in s and not s.upper().endswith("/USD")]
        if non_usdt:
            self._primary.prepare(non_usdt)
        if usdt:
            self._binance.prepare(usdt)

    def __call__(self, symbol: str) -> dict:
        if "/" in symbol and not symbol.upper().endswith("/USD"):
            return self._binance(symbol)
        return self._primary(symbol)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _build_binance_client_from_cfg(cfg: dict) -> Any | None:
    """Build a python-binance Client from the full application config dict.

    Returns the first enabled Binance account's Client, or None if no Binance
    config exists or the python-binance package is not installed.
    """
    try:
        from binance.client import Client  # type: ignore[import]
    except ImportError:
        log.debug("python-binance not installed; Binance market data unavailable")
        return None

    try:
        from app.brokers.config_utils import iter_binance_accounts
        accounts = list(iter_binance_accounts(cfg))
        if not accounts:
            return None
        acct = accounts[0]
        api_key = acct.get("api_key", "")
        api_secret = acct.get("api_secret", "")
        if not api_key or not api_secret:
            return None
        base_url = str(acct.get("base_url", "") or "")
        demo = bool(base_url and "demo" in base_url.lower())
        testnet = bool(acct.get("testnet", False))
        client = Client(api_key, api_secret, testnet=testnet, demo=demo)
        log.info("BinanceMarketDataProvider: client ready (demo=%s)", demo)
        return client
    except Exception as exc:
        log.warning("Failed to build Binance client for market data: %s", exc)
        return None

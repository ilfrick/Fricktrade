# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None


@dataclass
class MarketCacheConfig:
    enabled: bool
    redis_url: str
    file_dir: str
    batch_size: int
    cache_only: bool
    ignore_staleness: bool
    filtered_symbols_enabled: bool


def interval_to_seconds(interval: str) -> int:
    interval = str(interval or "").lower()
    if interval.endswith("m"):
        return max(int(interval[:-1]), 1) * 60
    if interval.endswith("h"):
        return max(int(interval[:-1]), 1) * 3600
    if interval.endswith("d"):
        return max(int(interval[:-1]), 1) * 86400
    return 60


def build_market_cache_config(cfg: dict | None) -> MarketCacheConfig:
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", False))
    return MarketCacheConfig(
        enabled=enabled,
        redis_url=str(cfg.get("redis_url", "redis://redis:6379/0")),
        file_dir=str(cfg.get("file_dir", "/data/market_cache")),
        batch_size=int(cfg.get("batch_size", 100)),
        cache_only=bool(cfg.get("cache_only", True)),
        ignore_staleness=bool(cfg.get("ignore_staleness", False)),
        filtered_symbols_enabled=bool(cfg.get("filtered_symbols", {}).get("enabled", True)),
    )


def build_market_cache(cfg: dict | None) -> "MarketCache | None":
    cache_cfg = build_market_cache_config(cfg)
    if not cache_cfg.enabled:
        return None
    return MarketCache(cache_cfg.redis_url, cache_cfg.file_dir, ignore_staleness=cache_cfg.ignore_staleness)


class MarketCache:
    def __init__(self, redis_url: str, file_dir: str, *, ignore_staleness: bool = False):
        self._redis = None
        if redis is not None:
            try:
                self._redis = redis.Redis.from_url(redis_url)
                self._redis.ping()
            except Exception as exc:
                logging.warning("Market cache redis unavailable: %s", exc)
                self._redis = None
        self._file_dir = Path(file_dir)
        self._bars_dir = self._file_dir / "bars"
        self._filtered_dir = self._file_dir / "filtered"
        self._ignore_staleness = ignore_staleness

    def get_bars(
        self,
        symbols: list[str],
        interval: str,
        *,
        max_age_seconds: int,
        lowercase: bool = False,
    ) -> dict[str, pd.DataFrame]:
        symbols = [s for s in symbols if s]
        if not symbols:
            return {}
        now = time.time()
        results: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        keys = [self._bars_key(interval, symbol) for symbol in symbols]
        if self._redis is not None:
            try:
                raw_values = self._redis.mget(keys)
                for symbol, raw in zip(symbols, raw_values):
                    if not raw:
                        missing.append(symbol)
                        continue
                    record = self._deserialize(raw)
                    if not record:
                        missing.append(symbol)
                        continue
                    if not self._ignore_staleness:
                        updated_at = record.get("updated_at", 0.0)
                        if now - updated_at > max_age_seconds:
                            missing.append(symbol)
                            continue
                    frame = record.get("data")
                    if isinstance(frame, pd.DataFrame):
                        results[symbol] = _normalize_frame(frame, lowercase=lowercase)
                if not missing:
                    return results
            except Exception as exc:
                logging.warning("Market cache redis read failed: %s", exc)
        for symbol in missing or symbols:
            frame = self._read_file(symbol, interval, max_age_seconds, ignore_staleness=self._ignore_staleness)
            if frame is not None:
                results[symbol] = _normalize_frame(frame, lowercase=lowercase)
        return results

    def set_bars(self, bars_by_symbol: dict[str, pd.DataFrame], interval: str, ttl_seconds: int) -> None:
        if not bars_by_symbol:
            return
        payloads = {}
        now = time.time()
        for symbol, frame in bars_by_symbol.items():
            if frame is None or frame.empty:
                continue
            record = {"updated_at": now, "data": frame}
            payloads[self._bars_key(interval, symbol)] = self._serialize(record)
            self._write_file(symbol, interval, record)
        if payloads and self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                for key, payload in payloads.items():
                    pipe.setex(key, ttl_seconds, payload)
                pipe.execute()
            except Exception as exc:
                logging.warning("Market cache redis write failed: %s", exc)

    def get_filtered_symbols(self, interval: str, max_age_seconds: int) -> list[str] | None:
        key = self._filtered_key(interval)
        now = time.time()
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw:
                    record = self._deserialize(raw)
                    if record:
                        if self._ignore_staleness:
                            symbols = record.get("symbols")
                            if isinstance(symbols, list):
                                return [str(s) for s in symbols if s]
                        else:
                            updated_at = record.get("updated_at", 0.0)
                            if now - updated_at <= max_age_seconds:
                                symbols = record.get("symbols")
                                if isinstance(symbols, list):
                                    return [str(s) for s in symbols if s]
            except Exception as exc:
                logging.warning("Market cache redis filtered read failed: %s", exc)
        return self._read_filtered_file(interval, max_age_seconds, ignore_staleness=self._ignore_staleness)

    def set_filtered_symbols(self, symbols: list[str], interval: str, ttl_seconds: int) -> None:
        symbols = [s for s in symbols if s]
        record = {"updated_at": time.time(), "symbols": symbols}
        payload = self._serialize(record)
        key = self._filtered_key(interval)
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl_seconds, payload)
            except Exception as exc:
                logging.warning("Market cache redis filtered write failed: %s", exc)
        self._write_filtered_file(interval, record)

    def _bars_key(self, interval: str, symbol: str) -> str:
        return f"market_cache:bars:{interval}:{symbol}"

    def _filtered_key(self, interval: str) -> str:
        return f"market_cache:filtered:{interval}"

    def _serialize(self, record: dict) -> bytes:
        return pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, payload: bytes) -> dict | None:
        try:
            record = pickle.loads(payload)
        except Exception:
            return None
        return record if isinstance(record, dict) else None

    def _read_file(
        self,
        symbol: str,
        interval: str,
        max_age_seconds: int,
        *,
        ignore_staleness: bool,
    ) -> pd.DataFrame | None:
        path = self._bars_dir / interval / f"{symbol}.pkl"
        if not path.exists():
            return None
        if not ignore_staleness and time.time() - path.stat().st_mtime > max_age_seconds:
            return None
        try:
            record = pd.read_pickle(path)
        except Exception:
            return None
        if isinstance(record, dict) and isinstance(record.get("data"), pd.DataFrame):
            return record["data"]
        if isinstance(record, pd.DataFrame):
            return record
        return None

    def _write_file(self, symbol: str, interval: str, record: dict) -> None:
        path = self._bars_dir / interval
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{symbol}.pkl"
        tmp = target.with_suffix(".tmp")
        try:
            pd.to_pickle(record, tmp)
            os.replace(tmp, target)
        except Exception as exc:
            logging.warning("Market cache file write failed for %s: %s", symbol, exc)

    def _read_filtered_file(
        self,
        interval: str,
        max_age_seconds: int,
        *,
        ignore_staleness: bool,
    ) -> list[str] | None:
        path = self._filtered_dir / f"{interval}.json"
        if not path.exists():
            return None
        if not ignore_staleness and time.time() - path.stat().st_mtime > max_age_seconds:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict):
            symbols = payload.get("symbols")
            if isinstance(symbols, list):
                return [str(s) for s in symbols if s]
        if isinstance(payload, list):
            return [str(s) for s in payload if s]
        return None

    def _write_filtered_file(self, interval: str, record: dict) -> None:
        path = self._filtered_dir
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{interval}.json"
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except Exception as exc:
            logging.warning("Market cache filtered file write failed: %s", exc)


def _normalize_frame(frame: pd.DataFrame, *, lowercase: bool) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    if lowercase:
        return frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
    if "close" in frame.columns:
        return frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
    return frame

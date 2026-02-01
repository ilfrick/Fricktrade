# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.monitoring.metrics import MARKET_CACHE_STALE_BARS, MARKET_CACHE_STALE_FILTERED

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
    delay_seconds: float
    max_age_multiplier: int
    cache_only: bool
    ignore_staleness: bool
    filtered_symbols_enabled: bool
    allow_pickle: bool


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
        delay_seconds=float(cfg.get("delay_seconds", 5.0)),
        max_age_multiplier=max(int(cfg.get("max_age_multiplier", 1)), 1),
        cache_only=bool(cfg.get("cache_only", True)),
        ignore_staleness=bool(cfg.get("ignore_staleness", False)),
        filtered_symbols_enabled=bool(cfg.get("filtered_symbols", {}).get("enabled", True)),
        allow_pickle=bool(cfg.get("allow_pickle", False)),
    )


def cache_max_age_seconds(interval: str, multiplier: int) -> int:
    base = interval_to_seconds(interval)
    mult = max(int(multiplier or 1), 1)
    return base * mult


def build_market_cache(cfg: dict | None) -> "MarketCache | None":
    cache_cfg = build_market_cache_config(cfg)
    if not cache_cfg.enabled:
        return None
    return MarketCache(
        cache_cfg.redis_url,
        cache_cfg.file_dir,
        ignore_staleness=cache_cfg.ignore_staleness,
        allow_pickle=cache_cfg.allow_pickle,
    )


class MarketCache:
    def __init__(
        self,
        redis_url: str,
        file_dir: str,
        *,
        ignore_staleness: bool = False,
        allow_pickle: bool = False,
    ):
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
        self._allow_pickle = allow_pickle

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
        stale_redis = 0
        stale_files = 0
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
                    updated_at = record.get("updated_at", 0.0)
                    if now - updated_at > max_age_seconds:
                        if self._ignore_staleness:
                            stale_redis += 1
                            MARKET_CACHE_STALE_BARS.labels(interval=interval, source="redis").inc()
                        else:
                            missing.append(symbol)
                            continue
                    frame = record.get("data")
                    if isinstance(frame, pd.DataFrame):
                        results[symbol] = _normalize_frame(frame, lowercase=lowercase)
                if not missing:
                    if self._ignore_staleness and stale_redis:
                        logging.warning(
                            "Market cache stale bars used; interval=%s stale_redis=%d stale_files=%d",
                            interval,
                            stale_redis,
                            stale_files,
                        )
                    return results
            except Exception as exc:
                logging.warning("Market cache redis read failed: %s", exc)
        for symbol in missing or symbols:
            json_path = self._bars_dir / interval / f"{symbol}.json"
            pkl_path = self._bars_dir / interval / f"{symbol}.pkl"
            path = None
            if json_path.exists():
                path = json_path
            elif self._allow_pickle and pkl_path.exists():
                path = pkl_path
            if self._ignore_staleness and path is not None:
                age = time.time() - path.stat().st_mtime
                if age > max_age_seconds:
                    stale_files += 1
                    MARKET_CACHE_STALE_BARS.labels(interval=interval, source="file").inc()
            frame = self._read_file(symbol, interval, max_age_seconds, ignore_staleness=self._ignore_staleness)
            if frame is not None:
                results[symbol] = _normalize_frame(frame, lowercase=lowercase)
        if self._ignore_staleness and (stale_redis or stale_files):
            logging.warning(
                "Market cache stale bars used; interval=%s stale_redis=%d stale_files=%d",
                interval,
                stale_redis,
                stale_files,
            )
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
        now = time.time()
        if self._redis is not None:
            try:
                prefix = self._filtered_symbol_key_prefix(interval)
                keys = list(self._redis.scan_iter(match=f"{prefix}*"))
                if keys:
                    symbols: list[str] = []
                    stale_count = 0
                    max_stale_age = 0.0
                    # Use mget for batch retrieval instead of N individual get calls
                    raw_values = self._redis.mget(keys)
                    for key, raw in zip(keys, raw_values):
                        if not raw:
                            continue
                        record = self._deserialize(raw) or {}
                        updated_at = float(record.get("updated_at", 0.0))
                        age = now - updated_at
                        symbol = self._filtered_symbol_from_key(key)
                        if not symbol:
                            continue
                        if self._ignore_staleness:
                            if age > max_age_seconds:
                                stale_count += 1
                                max_stale_age = max(max_stale_age, age)
                            symbols.append(symbol)
                        elif age <= max_age_seconds:
                            symbols.append(symbol)
                    if stale_count:
                        logging.warning(
                            "Market cache stale filtered symbols used; interval=%s stale=%d max_age_sec=%d",
                            interval,
                            stale_count,
                            int(max_stale_age),
                        )
                        MARKET_CACHE_STALE_FILTERED.labels(interval=interval, source="redis").inc(stale_count)
                    return symbols or None

                # Legacy single-blob fallback.
                key = self._filtered_key(interval)
                raw = self._redis.get(key)
                if raw:
                    record = self._deserialize(raw)
                    if record:
                        updated_at = record.get("updated_at", 0.0)
                        if self._ignore_staleness:
                            if now - updated_at > max_age_seconds:
                                logging.warning(
                                    "Market cache stale filtered symbols used; interval=%s age_sec=%d",
                                    interval,
                                    int(now - updated_at),
                                )
                                MARKET_CACHE_STALE_FILTERED.labels(interval=interval, source="redis").inc()
                            symbols = record.get("symbols")
                            if isinstance(symbols, list):
                                return [str(s) for s in symbols if s]
                        elif now - updated_at <= max_age_seconds:
                            symbols = record.get("symbols")
                            if isinstance(symbols, list):
                                return [str(s) for s in symbols if s]
            except Exception as exc:
                logging.warning("Market cache redis filtered read failed: %s", exc)
        per_symbol_dir = self._filtered_dir / interval
        if per_symbol_dir.exists():
            symbols: list[str] = []
            stale_count = 0
            max_stale_age = 0.0
            for path in per_symbol_dir.glob("*.json"):
                symbol = path.stem
                updated_at = None
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        updated_at = payload.get("updated_at")
                except Exception:
                    updated_at = None
                if updated_at is None:
                    age = time.time() - path.stat().st_mtime
                else:
                    age = now - float(updated_at)
                if self._ignore_staleness:
                    if age > max_age_seconds:
                        stale_count += 1
                        max_stale_age = max(max_stale_age, age)
                    symbols.append(symbol)
                elif age <= max_age_seconds:
                    symbols.append(symbol)
            if stale_count:
                logging.warning(
                    "Market cache stale filtered symbols used; interval=%s stale=%d max_age_sec=%d",
                    interval,
                    stale_count,
                    int(max_stale_age),
                )
                MARKET_CACHE_STALE_FILTERED.labels(interval=interval, source="file").inc(stale_count)
            if symbols:
                return symbols

        # Legacy single-blob fallback.
        path = self._filtered_dir / f"{interval}.json"
        if self._ignore_staleness and path.exists():
            age = time.time() - path.stat().st_mtime
            if age > max_age_seconds:
                logging.warning(
                    "Market cache stale filtered symbols used; interval=%s age_sec=%d",
                    interval,
                    int(age),
                )
                MARKET_CACHE_STALE_FILTERED.labels(interval=interval, source="file").inc()
        return self._read_filtered_file(interval, max_age_seconds, ignore_staleness=self._ignore_staleness)

    def set_filtered_symbols(self, symbols: list[str], interval: str, ttl_seconds: int) -> None:
        symbols = [s for s in symbols if s]
        record = {"updated_at": time.time()}
        payload = self._serialize(record)
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                for symbol in symbols:
                    pipe.setex(self._filtered_symbol_key(interval, symbol), ttl_seconds, payload)
                pipe.execute()
            except Exception as exc:
                logging.warning("Market cache redis filtered write failed: %s", exc)
        for symbol in symbols:
            self._write_filtered_symbol_file(interval, symbol, record)

    def _sanitize_symbol(self, symbol: str) -> str:
        """Sanitize symbol for use in Redis keys (replace special chars)."""
        return symbol.replace(":", "_").replace("*", "_").replace("?", "_")

    def _bars_key(self, interval: str, symbol: str) -> str:
        return f"market_cache:bars:{interval}:{self._sanitize_symbol(symbol)}"

    def _filtered_key(self, interval: str) -> str:
        return f"market_cache:filtered:{interval}"

    def _filtered_symbol_key_prefix(self, interval: str) -> str:
        return f"{self._filtered_key(interval)}:"

    def _filtered_symbol_key(self, interval: str, symbol: str) -> str:
        return f"{self._filtered_key(interval)}:{self._sanitize_symbol(symbol)}"

    def _filtered_symbol_from_key(self, key: object) -> str | None:
        if key is None:
            return None
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="ignore")
        if not isinstance(key, str):
            return None
        parts = key.split(":")
        if len(parts) < 4:
            return None
        return parts[-1] or None

    def _encode_record(self, record: dict) -> dict:
        if not isinstance(record, dict):
            return {}
        payload = dict(record)
        data = payload.get("data")
        if isinstance(data, pd.DataFrame):
            payload["data"] = data.to_json(orient="split", date_format="iso")
            payload["data_format"] = "split-json"
        return payload

    def _decode_record(self, record: object) -> dict | None:
        if not isinstance(record, dict):
            return None
        data = record.get("data")
        if isinstance(data, str) and record.get("data_format") == "split-json":
            try:
                frame = pd.read_json(io.StringIO(data), orient="split", convert_dates=["index"])
            except Exception:
                return None
            decoded = dict(record)
            decoded["data"] = frame
            return decoded
        return record

    def _serialize(self, record: dict) -> bytes:
        payload = self._encode_record(record)
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _deserialize(self, payload: bytes) -> dict | None:
        try:
            text = payload.decode("utf-8")
            record = json.loads(text)
            return self._decode_record(record)
        except Exception:
            if not self._allow_pickle:
                return None
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
        json_path = self._bars_dir / interval / f"{symbol}.json"
        if json_path.exists():
            if not ignore_staleness and time.time() - json_path.stat().st_mtime > max_age_seconds:
                return None
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                return None
            record = self._decode_record(payload)
            if record and isinstance(record.get("data"), pd.DataFrame):
                return record["data"]
        if not self._allow_pickle:
            return None
        pkl_path = self._bars_dir / interval / f"{symbol}.pkl"
        if not pkl_path.exists():
            return None
        if not ignore_staleness and time.time() - pkl_path.stat().st_mtime > max_age_seconds:
            return None
        try:
            record = pd.read_pickle(pkl_path)
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
        target = path / f"{symbol}.json"
        tmp = target.with_suffix(".tmp")
        try:
            payload = self._encode_record(record)
            tmp.write_text(json.dumps(payload), encoding="utf-8")
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

    def _write_filtered_symbol_file(self, interval: str, symbol: str, record: dict) -> None:
        path = self._filtered_dir / interval
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{symbol}.json"
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except Exception as exc:
            logging.warning("Market cache filtered symbol file write failed for %s: %s", symbol, exc)


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

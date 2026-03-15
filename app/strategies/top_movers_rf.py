# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import logging
import pickle
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.strategies.base import Strategy

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover - optional dependency in some runtimes
    RandomForestClassifier = None


_FILE_RE = re.compile(r"^(?P<symbol>.+)_(?P<day>\d{4}-\d{2}-\d{2})_1m\.csv$")
_BUNDLE_VERSION = 1

_NOWCAST_FEATURES = [
    "bars_seen",
    "ret_from_open_pct",
    "runup_pct",
    "drawdown_pct",
    "range_pct",
    "ret_std_pct",
    "ret_mean_pct",
    "vol_sum",
    "vol_per_bar",
    "vwap_dist_pct",
    "trend_slope",
    "pos_in_range",
]

_ENTRY_FEATURES = [
    "ret_from_open_pct",
    "pullback_from_high_pct",
    "dist_to_low_so_far_pct",
    "ret_std_pct",
    "mom_5_pct",
    "mom_20_pct",
    "vol_spike",
    "vwap_dist_pct",
    "bars_seen",
    "pos_in_range",
]


def _interval_minutes(interval: str) -> int:
    text = str(interval or "1m").strip().lower()
    if text.endswith("m"):
        return max(int(text[:-1] or "1"), 1)
    if text.endswith("h"):
        return max(int(text[:-1] or "1") * 60, 1)
    return 1


def _bars_from_minutes(minutes: int, interval: str) -> int:
    mins = _interval_minutes(interval)
    return max(int(round(float(minutes) / float(mins))), 1)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _as_float_list(values) -> list[float]:
    out: list[float] = []
    for value in values or []:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"datetime", "timestamp", "date", "time"}:
            col_map[c] = "datetime"
        elif cl in {"open", "o"}:
            col_map[c] = "open"
        elif cl in {"high", "h"}:
            col_map[c] = "high"
        elif cl in {"low", "l"}:
            col_map[c] = "low"
        elif cl in {"close", "c"}:
            col_map[c] = "close"
        elif cl in {"volume", "v"}:
            col_map[c] = "volume"

    if not {"open", "high", "low", "close"}.issubset(set(col_map.values())):
        return None
    out = df.rename(columns=col_map).copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)
        out = out.sort_values("datetime")
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if out.empty:
        return None
    return out


def _resample_frame(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    mins = _interval_minutes(interval)
    if mins <= 1 or len(df) <= 1:
        return df
    groups = np.arange(len(df), dtype=int) // mins
    frame = pd.DataFrame(
        {
            "open": df["open"].groupby(groups).first(),
            "high": df["high"].groupby(groups).max(),
            "low": df["low"].groupby(groups).min(),
            "close": df["close"].groupby(groups).last(),
            "volume": df["volume"].groupby(groups).sum(),
        }
    )
    if "datetime" in df.columns:
        frame["datetime"] = df["datetime"].groupby(groups).last().values
    return frame.reset_index(drop=True)


def _nowcast_features_from_arrays(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    cutoff_bars: int,
) -> dict[str, float]:
    n = max(min(cutoff_bars, int(closes.size)), 1)
    c0 = closes[:n]
    h0 = highs[:n] if highs.size >= n else c0
    l0 = lows[:n] if lows.size >= n else c0
    v0 = volumes[:n] if volumes.size >= n else np.zeros_like(c0)

    open_px = float(c0[0])
    last_px = float(c0[-1])
    high_px = float(np.max(h0))
    low_px = float(np.min(l0))
    vol_sum = float(np.sum(v0))

    ret = np.diff(c0) / np.maximum(c0[:-1], 1e-9)
    ret_std = float(np.std(ret) * 100.0) if ret.size else 0.0
    ret_mean = float(np.mean(ret) * 100.0) if ret.size else 0.0

    vwap_dist = 0.0
    if vol_sum > 0:
        vwap = float(np.sum(c0 * v0) / vol_sum)
        if vwap > 0:
            vwap_dist = (last_px - vwap) / vwap * 100.0

    idx = np.arange(c0.size, dtype=float)
    norm = c0 / max(open_px, 1e-9)
    slope = float(np.polyfit(idx, norm, 1)[0]) if c0.size >= 3 else 0.0

    pos_in_range = 0.5
    if high_px > low_px:
        pos_in_range = (last_px - low_px) / (high_px - low_px)

    return {
        "bars_seen": int(n),
        "ret_from_open_pct": (last_px - open_px) / max(open_px, 1e-9) * 100.0,
        "runup_pct": (high_px - open_px) / max(open_px, 1e-9) * 100.0,
        "drawdown_pct": (low_px - open_px) / max(open_px, 1e-9) * 100.0,
        "range_pct": (high_px - low_px) / max(open_px, 1e-9) * 100.0,
        "ret_std_pct": ret_std,
        "ret_mean_pct": ret_mean,
        "vol_sum": vol_sum,
        "vol_per_bar": vol_sum / max(float(v0.size), 1.0),
        "vwap_dist_pct": vwap_dist,
        "trend_slope": slope,
        "pos_in_range": float(pos_in_range),
    }


def _entry_features_from_arrays(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, float]:
    eps = 1e-9
    current = float(closes[-1])
    open_px = float(closes[0])
    high_so_far = float(np.max(highs)) if highs.size else current
    low_so_far = float(np.min(lows)) if lows.size else current
    past_v = volumes if volumes.size else np.zeros_like(closes)

    ret = np.diff(closes) / np.maximum(closes[:-1], eps)
    ret_std = float(np.std(ret) * 100.0) if ret.size else 0.0

    vol_recent = float(np.sum(past_v[-5:])) if past_v.size else 0.0
    vol_avg20 = float(np.mean(past_v[-20:])) if past_v.size >= 20 else (float(np.mean(past_v)) if past_v.size else 0.0)
    vol_spike = vol_recent / max(vol_avg20 * 5.0, eps)

    mom_5 = 0.0
    mom_20 = 0.0
    if closes.size >= 6:
        mom_5 = (current - float(closes[-6])) / max(current, eps) * 100.0
    if closes.size >= 21:
        mom_20 = (current - float(closes[-21])) / max(current, eps) * 100.0

    vwap_dist = 0.0
    vol_sum = float(np.sum(past_v)) if past_v.size else 0.0
    if vol_sum > 0:
        vwap = float(np.sum(closes * past_v) / vol_sum)
        if vwap > 0:
            vwap_dist = (current - vwap) / vwap * 100.0

    pos_in_range = 0.5
    if high_so_far > low_so_far:
        pos_in_range = (current - low_so_far) / (high_so_far - low_so_far)

    return {
        "ret_from_open_pct": (current - open_px) / max(open_px, eps) * 100.0,
        "pullback_from_high_pct": (current - high_so_far) / max(high_so_far, eps) * 100.0,
        "dist_to_low_so_far_pct": (current - low_so_far) / max(low_so_far, eps) * 100.0,
        "ret_std_pct": ret_std,
        "mom_5_pct": mom_5,
        "mom_20_pct": mom_20,
        "vol_spike": vol_spike,
        "vwap_dist_pct": vwap_dist,
        "bars_seen": int(closes.size),
        "pos_in_range": float(pos_in_range),
    }


def _predict_proba(model, row: dict[str, float], features: list[str]) -> float:
    frame = pd.DataFrame([{c: float(row.get(c, 0.0) or 0.0) for c in features}], columns=features)
    proba = model.predict_proba(frame)
    if getattr(proba, "shape", (0, 0))[1] >= 2:
        return float(proba[0, 1])
    return float(proba[0, 0])


class _ConstantProbaModel:
    def __init__(self, positive_prob: float):
        p = float(positive_prob)
        self._p = max(min(p, 1.0), 0.0)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        n = int(len(frame))
        p = np.full((n,), self._p, dtype=float)
        return np.column_stack([1.0 - p, p])


class TopMoversRFStrategy(Strategy):
    _bundle_cache: dict[str, tuple[dict | None, float | None]] = {}
    _bundle_lock = threading.Lock()

    def __init__(self, cfg: dict | None = None, runtime_interval: str = "1m"):
        self._cfg = cfg or {}
        self._runtime_interval = str(self._cfg.get("runtime_interval") or runtime_interval or "1m")
        self._training_interval = str(self._cfg.get("training_interval") or self._runtime_interval)

        self._data_dir = Path(str(self._cfg.get("data_dir", "/data")))
        self._model_path = Path(str(self._cfg.get("model_path", "/data/models/top_movers_rf/top_movers_rf.pkl")))
        self._auto_train = _bool(self._cfg.get("auto_train_on_start", True), default=True)
        self._max_age_hours = _to_float(self._cfg.get("retrain_if_older_hours", 24), 24.0)
        self._min_training_days = max(_to_int(self._cfg.get("min_training_days", 5), 5), 2)
        self._min_bars = max(_to_int(self._cfg.get("min_bars", 60), 60), 20)
        self._top_n = max(_to_int(self._cfg.get("top_n", 10), 10), 1)
        self._random_state = _to_int(self._cfg.get("random_state", 42), 42)

        self._cutoff_minutes = max(_to_int(self._cfg.get("cutoff_minutes", 30), 30), 5)
        self._entry_horizon_minutes = max(_to_int(self._cfg.get("entry_horizon_minutes", 90), 90), 5)
        self._entry_warmup_minutes = max(_to_int(self._cfg.get("entry_warmup_minutes", 20), 20), 5)
        self._low_zone_tol_pct = max(_to_float(self._cfg.get("low_zone_tol_pct", 0.35), 0.35), 0.0)
        self._rebound_target_pct = max(_to_float(self._cfg.get("rebound_target_pct", 2.0), 2.0), 0.1)

        self._buy_nowcast_min = _to_float(self._cfg.get("buy_nowcast_min", 0.55), 0.55)
        self._buy_entry_min = _to_float(self._cfg.get("buy_entry_min", 0.55), 0.55)
        self._buy_score_min = _to_float(self._cfg.get("buy_score_min", 0.58), 0.58)
        self._exit_score_max = _to_float(self._cfg.get("exit_score_max", 0.40), 0.40)
        self._exit_pullback_pct = _to_float(self._cfg.get("exit_pullback_from_high_pct", 1.2), 1.2)

        self._nowcast_weight = max(_to_float(self._cfg.get("nowcast_weight", 0.55), 0.55), 0.0)
        self._entry_weight = max(_to_float(self._cfg.get("entry_weight", 0.45), 0.45), 0.0)
        if self._nowcast_weight + self._entry_weight <= 0:
            self._nowcast_weight = 0.5
            self._entry_weight = 0.5

        self._respect_account_blocks = _bool(self._cfg.get("respect_account_blocks", True), default=True)
        self._pdt_soft_block = _bool(self._cfg.get("respect_pdt_soft_block", True), default=True)
        self._pdt_daytrade_limit = max(_to_int(self._cfg.get("pdt_daytrade_limit", 3), 3), 0)
        self._pdt_min_equity = max(_to_float(self._cfg.get("pdt_min_equity", 25000.0), 25000.0), 0.0)
        self._min_buying_power = max(_to_float(self._cfg.get("min_buying_power", 50.0), 50.0), 0.0)

        self._bundle = self._load_or_train_bundle()

    def _cache_key(self) -> str:
        try:
            return str(self._model_path.resolve())
        except OSError:
            return str(self._model_path)

    def _load_or_train_bundle(self) -> dict | None:
        cache_key = self._cache_key()
        file_mtime = self._model_mtime()
        with self._bundle_lock:
            cached = self._bundle_cache.get(cache_key)
            if cached and cached[1] == file_mtime:
                return cached[0]

        bundle = self._load_bundle_file()
        if bundle is None and self._auto_train:
            bundle = self._train_bundle_and_save()
            file_mtime = self._model_mtime()
        elif bundle is not None and self._auto_train and self._is_bundle_stale():
            bundle = self._train_bundle_and_save()
            file_mtime = self._model_mtime()

        with self._bundle_lock:
            self._bundle_cache[cache_key] = (bundle, file_mtime)
        return bundle

    def _model_mtime(self) -> float | None:
        try:
            return self._model_path.stat().st_mtime
        except OSError:
            return None

    def _is_bundle_stale(self) -> bool:
        if self._max_age_hours <= 0:
            return False
        try:
            mtime = self._model_path.stat().st_mtime
        except OSError:
            return True
        age_seconds = datetime.now(timezone.utc).timestamp() - float(mtime)
        return age_seconds > float(self._max_age_hours) * 3600.0

    def _load_bundle_file(self) -> dict | None:
        if not self._model_path.exists():
            return None
        try:
            with self._model_path.open("rb") as handle:
                bundle = pickle.load(handle)
        except Exception as exc:
            logging.warning("top_movers_rf: failed to load model bundle %s: %s", self._model_path, exc)
            return None
        if not isinstance(bundle, dict):
            return None
        if int(bundle.get("bundle_version", 0) or 0) != _BUNDLE_VERSION:
            return None
        models = bundle.get("models") or {}
        features = bundle.get("features") or {}
        if not isinstance(models.get("nowcast"), object) or not isinstance(models.get("entry"), object):
            return None
        if not isinstance(features.get("nowcast"), list) or not isinstance(features.get("entry"), list):
            return None
        return bundle

    def _write_bundle_file(self, bundle: dict) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._model_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self._model_path)

    def _train_bundle_and_save(self) -> dict | None:
        bundle = self._train_bundle()
        if bundle is None:
            return None
        try:
            self._write_bundle_file(bundle)
        except OSError as exc:
            logging.warning("top_movers_rf: failed to persist model bundle: %s", exc)
        return bundle

    def _train_bundle(self) -> dict | None:
        if RandomForestClassifier is None:
            logging.warning("top_movers_rf: scikit-learn unavailable; cannot train model.")
            return None

        cutoff_bars = _bars_from_minutes(self._cutoff_minutes, self._training_interval)
        warmup_bars = _bars_from_minutes(self._entry_warmup_minutes, self._training_interval)
        horizon_bars = _bars_from_minutes(self._entry_horizon_minutes, self._training_interval)
        min_bars = max(self._min_bars, cutoff_bars + 1, warmup_bars + horizon_bars + 1)

        symbol_days = self._load_symbol_days(min_bars=min_bars)
        if not symbol_days:
            logging.warning("top_movers_rf: no training symbol/day rows found under %s", self._data_dir)
            return None

        unique_days = sorted({row["date"] for row in symbol_days})
        if len(unique_days) < self._min_training_days:
            logging.warning(
                "top_movers_rf: insufficient training days (%d < %d)",
                len(unique_days),
                self._min_training_days,
            )
            return None

        nowcast_df = self._build_nowcast_dataset(symbol_days, cutoff_bars)
        entry_df = self._build_entry_dataset(symbol_days, warmup_bars, horizon_bars)
        if nowcast_df.empty or entry_df.empty:
            logging.warning("top_movers_rf: training datasets are empty (nowcast=%d entry=%d)", len(nowcast_df), len(entry_df))
            return None

        y_nowcast = nowcast_df["target_top_n"].astype(int)
        y_entry = entry_df["target_entry"].astype(int)
        if y_nowcast.nunique() < 2:
            nowcast_model = _ConstantProbaModel(float(y_nowcast.mean()))
            logging.warning(
                "top_movers_rf: nowcast labels have no class variance; using constant model (p=%.4f)",
                float(y_nowcast.mean()),
            )
        else:
            nowcast_model = RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=self._random_state,
                n_jobs=-1,
            )
            nowcast_model.fit(nowcast_df[_NOWCAST_FEATURES], y_nowcast)

        if y_entry.nunique() < 2:
            entry_model = _ConstantProbaModel(float(y_entry.mean()))
            logging.warning(
                "top_movers_rf: entry labels have no class variance; using constant model (p=%.4f)",
                float(y_entry.mean()),
            )
        else:
            entry_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=self._random_state,
                n_jobs=-1,
            )
            entry_model.fit(entry_df[_ENTRY_FEATURES], y_entry)

        bundle = {
            "bundle_version": _BUNDLE_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": {
                "nowcast": nowcast_model,
                "entry": entry_model,
            },
            "features": {
                "nowcast": list(_NOWCAST_FEATURES),
                "entry": list(_ENTRY_FEATURES),
            },
            "training": {
                "data_dir": str(self._data_dir),
                "training_interval": self._training_interval,
                "runtime_interval": self._runtime_interval,
                "days": int(len(unique_days)),
                "symbol_days": int(len(symbol_days)),
                "nowcast_rows": int(len(nowcast_df)),
                "entry_rows": int(len(entry_df)),
                "top_n": int(self._top_n),
                "cutoff_minutes": int(self._cutoff_minutes),
                "entry_horizon_minutes": int(self._entry_horizon_minutes),
                "entry_warmup_minutes": int(self._entry_warmup_minutes),
                "low_zone_tol_pct": float(self._low_zone_tol_pct),
                "rebound_target_pct": float(self._rebound_target_pct),
                "date_min": str(unique_days[0]),
                "date_max": str(unique_days[-1]),
            },
        }
        logging.info(
            "top_movers_rf: trained models with %d days, %d nowcast rows, %d entry rows",
            len(unique_days),
            len(nowcast_df),
            len(entry_df),
        )
        return bundle

    def _load_symbol_days(self, min_bars: int) -> list[dict]:
        rows: list[dict] = []
        for path in sorted(self._data_dir.glob("*_????-??-??_1m.csv")):
            match = _FILE_RE.match(path.name)
            if not match:
                continue
            day = date.fromisoformat(match.group("day"))
            symbol = match.group("symbol")
            try:
                raw = pd.read_csv(path)
            except Exception:
                continue
            normalized = _normalize_ohlcv(raw)
            if normalized is None or normalized.empty:
                continue
            normalized = _resample_frame(normalized, self._training_interval)
            if len(normalized) < min_bars:
                continue
            rows.append({"symbol": symbol, "date": day, "frame": normalized})
        return rows

    def _build_nowcast_dataset(self, symbol_days: list[dict], cutoff_bars: int) -> pd.DataFrame:
        per_day: dict[date, list[dict]] = {}
        for row in symbol_days:
            c = row["frame"]["close"].to_numpy(dtype=float)
            day_ret = (float(c[-1]) - float(c[0])) / max(float(c[0]), 1e-9) * 100.0
            per_day.setdefault(row["date"], []).append({"symbol": row["symbol"], "day_return_pct": day_ret})

        top_by_day: dict[date, set[str]] = {}
        for d, entries in per_day.items():
            ranked = sorted(entries, key=lambda x: x["day_return_pct"], reverse=True)
            top_by_day[d] = {entry["symbol"] for entry in ranked[: self._top_n]}

        out: list[dict] = []
        for row in symbol_days:
            frame = row["frame"]
            closes = frame["close"].to_numpy(dtype=float)
            highs = frame["high"].to_numpy(dtype=float)
            lows = frame["low"].to_numpy(dtype=float)
            volumes = frame["volume"].to_numpy(dtype=float)
            feats = _nowcast_features_from_arrays(closes, highs, lows, volumes, cutoff_bars)
            out.append(
                {
                    "date": row["date"],
                    "symbol": row["symbol"],
                    "target_top_n": 1 if row["symbol"] in top_by_day.get(row["date"], set()) else 0,
                    **feats,
                }
            )
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out).sort_values(["date", "symbol"]).reset_index(drop=True)

    def _build_entry_dataset(
        self,
        symbol_days: list[dict],
        warmup_bars: int,
        horizon_bars: int,
    ) -> pd.DataFrame:
        out: list[dict] = []
        eps = 1e-9
        for row in symbol_days:
            frame = row["frame"]
            closes = frame["close"].to_numpy(dtype=float)
            highs = frame["high"].to_numpy(dtype=float)
            lows = frame["low"].to_numpy(dtype=float)
            volumes = frame["volume"].to_numpy(dtype=float)
            if closes.size <= warmup_bars + horizon_bars:
                continue
            for t in range(warmup_bars, int(closes.size) - horizon_bars):
                current = float(closes[t])
                if current <= 0:
                    continue
                future_h = highs[t + 1 : t + 1 + horizon_bars]
                future_max = float(np.max(future_h)) if future_h.size else current
                # Use rolling 30-bar local low instead of full-day low to avoid
                # look-ahead bias (the full-day low is known only at session end).
                _lookback = 30
                local_lows = lows[max(0, t - _lookback) : t + 1]
                local_low = float(np.min(local_lows)) if local_lows.size else current
                dist_to_day_low_pct = (current - local_low) / max(local_low, eps) * 100.0
                future_rebound_pct = (future_max - current) / max(current, eps) * 100.0
                target = 1 if (
                    dist_to_day_low_pct <= self._low_zone_tol_pct
                    and future_rebound_pct >= self._rebound_target_pct
                ) else 0

                feats = _entry_features_from_arrays(
                    closes[: t + 1],
                    highs[: t + 1],
                    lows[: t + 1],
                    volumes[: t + 1],
                )
                out.append(
                    {
                        "date": row["date"],
                        "symbol": row["symbol"],
                        "target_entry": int(target),
                        **feats,
                    }
                )
        if not out:
            return pd.DataFrame()
        return pd.DataFrame(out).sort_values(["date", "symbol"]).reset_index(drop=True)

    def _session_arrays(self, market_state: dict) -> dict[str, np.ndarray]:
        prices = _as_float_list(market_state.get("session_prices") or market_state.get("prices") or [])
        highs = _as_float_list(market_state.get("session_highs") or market_state.get("highs") or prices)
        lows = _as_float_list(market_state.get("session_lows") or market_state.get("lows") or prices)
        volumes = _as_float_list(market_state.get("session_volumes") or market_state.get("volumes") or [])
        if not volumes:
            volumes = [0.0] * len(prices)
        if len(highs) < len(prices):
            highs = prices[:]
        if len(lows) < len(prices):
            lows = prices[:]
        if len(volumes) < len(prices):
            volumes = volumes + [0.0] * (len(prices) - len(volumes))
        return {
            "prices": np.asarray(prices, dtype=float),
            "highs": np.asarray(highs, dtype=float),
            "lows": np.asarray(lows, dtype=float),
            "volumes": np.asarray(volumes, dtype=float),
        }

    def _is_account_blocked(self, market_state: dict) -> bool:
        if not self._respect_account_blocks:
            return False
        flags = market_state.get("account_flags") or {}
        for key in ("account_blocked", "trading_blocked", "trade_suspended_by_user"):
            if _bool(flags.get(key), default=False):
                return True
        return False

    def _is_pdt_soft_block(self, market_state: dict) -> bool:
        if not self._pdt_soft_block:
            return False
        flags = market_state.get("account_flags") or {}
        is_pdt = _bool(flags.get("pattern_day_trader"), default=False)
        if not is_pdt:
            return False
        daytrade_count = _to_float(flags.get("daytrade_count"), 0.0)
        equity = _to_float((market_state.get("portfolio") or {}).get("equity"), 0.0)
        return daytrade_count >= float(self._pdt_daytrade_limit) and equity < float(self._pdt_min_equity)

    def _has_buying_power(self, market_state: dict) -> bool:
        portfolio = market_state.get("portfolio") or {}
        buying_power = _to_float(portfolio.get("buying_power"), 0.0)
        cash = _to_float(portfolio.get("cash"), 0.0)
        available = buying_power if buying_power > 0 else cash
        return available >= self._min_buying_power

    def generate_signal(self, market_state: dict) -> dict:
        symbol = str(market_state.get("symbol", "") or "")
        if "/" in symbol:
            return {"action": "hold", "confidence": 0.0, "reason": "equity_only"}

        if not self._bundle:
            return {"action": "hold", "reason": "model_unavailable"}

        if self._is_account_blocked(market_state):
            return {"action": "hold", "reason": "account_blocked"}

        portfolio = market_state.get("portfolio") or {}
        symbol = str(market_state.get("symbol", "") or "")
        positions = portfolio.get("positions") if isinstance(portfolio, dict) else {}
        position_qty = _to_float((positions or {}).get(symbol, {}).get("qty"), 0.0)
        if position_qty <= 0 and self._is_pdt_soft_block(market_state):
            return {"action": "hold", "reason": "pdt_soft_block"}

        session = self._session_arrays(market_state)
        prices = session["prices"]
        if prices.size < 2:
            return {"action": "hold", "reason": "insufficient_session_data"}

        cutoff_bars = _bars_from_minutes(self._cutoff_minutes, self._runtime_interval)
        warmup_bars = _bars_from_minutes(self._entry_warmup_minutes, self._runtime_interval)
        if prices.size < cutoff_bars:
            return {"action": "hold", "reason": "insufficient_cutoff_bars"}

        nowcast_row = _nowcast_features_from_arrays(
            prices,
            session["highs"],
            session["lows"],
            session["volumes"],
            cutoff_bars,
        )
        nowcast_score = _predict_proba(
            self._bundle["models"]["nowcast"],
            nowcast_row,
            self._bundle["features"]["nowcast"],
        )

        entry_score = 0.0
        if prices.size >= warmup_bars:
            entry_row = _entry_features_from_arrays(
                prices,
                session["highs"],
                session["lows"],
                session["volumes"],
            )
            # Clip bars_seen to training range to avoid distribution shift:
            # training used 0..cutoff_bars; inference accumulates unboundedly.
            if "bars_seen" in entry_row:
                entry_row["bars_seen"] = min(entry_row["bars_seen"], cutoff_bars)
            entry_score = _predict_proba(
                self._bundle["models"]["entry"],
                entry_row,
                self._bundle["features"]["entry"],
            )

        total_weight = self._nowcast_weight + self._entry_weight
        score = (
            (self._nowcast_weight * nowcast_score) + (self._entry_weight * entry_score)
        ) / max(total_weight, 1e-9)

        signal = {
            "action": "hold",
            "confidence": float(max(min(score, 1.0), 0.0)),
            "score": float(score),
            "nowcast_score": float(nowcast_score),
            "entry_score": float(entry_score),
            "reason": "below_threshold",
        }

        if position_qty > 0:
            high_so_far = float(np.max(session["highs"])) if session["highs"].size else float(prices[-1])
            pullback_pct = ((high_so_far - float(prices[-1])) / max(high_so_far, 1e-9)) * 100.0
            signal["pullback_from_high_pct"] = float(pullback_pct)
            if score <= self._exit_score_max or pullback_pct >= self._exit_pullback_pct:
                signal["action"] = "exit"
                signal["confidence"] = float(max(min(1.0 - score, 1.0), 0.0))
                signal["reason"] = "model_exit"
            else:
                signal["reason"] = "position_hold"
            return signal

        if not self._has_buying_power(market_state):
            signal["reason"] = "insufficient_buying_power"
            return signal

        if (
            nowcast_score >= self._buy_nowcast_min
            and entry_score >= self._buy_entry_min
            and score >= self._buy_score_min
        ):
            signal["action"] = "buy"
            signal["reason"] = "model_entry"
        return signal

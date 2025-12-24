from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import logging
import math

import numpy as np
import pandas as pd
import torch
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


@dataclass
class AISymbolFilterConfig:
    interval: str
    lookback_days: int
    window: int
    retrain_hours: int
    model_path: str
    train_max_symbols: int
    max_samples_per_symbol: int
    timeout_seconds: int
    retries: int
    objective: str
    feed: str


def score_symbols(
    symbols: Iterable[str],
    api_key: str,
    api_secret: str,
    cfg: dict,
) -> tuple[list[str], dict[str, float]]:
    symbols = [s for s in symbols if s]
    if not symbols or not api_key or not api_secret:
        return [], {}
    config = _read_config(cfg)
    model_path = Path(config.model_path)
    model, stats = _load_model(model_path, config.retrain_hours)
    if model is None or stats is None:
        model, stats = _train_model(symbols, api_key, api_secret, config)
        if model is not None and stats is not None:
            _save_model(model_path, model, stats)

    if model is None or stats is None:
        return symbols, {s: 0.0 for s in symbols}

    bars = _fetch_bars(symbols, api_key, api_secret, config, limit_symbols=None)
    scores = {}
    for symbol, frame in bars.items():
        features = _latest_features(frame, config.window)
        if features is None:
            scores[symbol] = 0.0
            continue
        score = _predict(model, stats, features)
        scores[symbol] = float(score)
    for symbol in symbols:
        scores.setdefault(symbol, 0.0)
    ordered = sorted(symbols, key=lambda s: scores.get(s, 0.0), reverse=True)
    return ordered, scores


def _read_config(cfg: dict) -> AISymbolFilterConfig:
    interval = str(cfg.get("interval", "5m"))
    lookback_days = int(cfg.get("lookback_days", 20))
    window = int(cfg.get("window", 20))
    retrain_hours = int(cfg.get("retrain_hours", 6))
    model_path = str(cfg.get("model_path", "/data/ai_symbol_filter.pt"))
    train_max_symbols = int(cfg.get("train_max_symbols", 300))
    max_samples_per_symbol = int(cfg.get("max_samples_per_symbol", 200))
    timeout_seconds = int(cfg.get("timeout_seconds", 15))
    retries = int(cfg.get("retries", 2))
    objective = str(cfg.get("objective", "return"))
    feed = str(cfg.get("feed", "iex"))
    return AISymbolFilterConfig(
        interval=interval,
        lookback_days=lookback_days,
        window=window,
        retrain_hours=retrain_hours,
        model_path=model_path,
        train_max_symbols=train_max_symbols,
        max_samples_per_symbol=max_samples_per_symbol,
        timeout_seconds=timeout_seconds,
        retries=retries,
        objective=objective,
        feed=feed,
    )


def _load_model(model_path: Path, retrain_hours: int):
    if not model_path.exists():
        return None, None
    try:
        payload = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        logging.warning("AI filter model load failed: %s", exc)
        return None, None
    trained_at = payload.get("trained_at")
    if trained_at:
        trained_dt = datetime.fromisoformat(trained_at)
        if datetime.utcnow() - trained_dt > timedelta(hours=retrain_hours):
            return None, None
    model = torch.nn.Linear(payload["input_dim"], 1)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    stats = {
        "mean": payload.get("mean"),
        "std": payload.get("std"),
        "objective": payload.get("objective"),
    }
    return model, stats


def _save_model(model_path: Path, model, stats: dict):
    payload = {
        "input_dim": model.in_features,
        "state_dict": model.state_dict(),
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "objective": stats.get("objective"),
        "trained_at": datetime.utcnow().isoformat(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)


def _train_model(symbols: list[str], api_key: str, api_secret: str, cfg: AISymbolFilterConfig):
    train_symbols = symbols[: cfg.train_max_symbols]
    bars = _fetch_bars(train_symbols, api_key, api_secret, cfg, limit_symbols=cfg.train_max_symbols)
    features, labels = _build_training_data(bars, cfg)
    if features.size == 0:
        logging.warning("AI filter training skipped: no data")
        return None, None
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    features = (features - mean) / std
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    model = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    model.eval()
    stats = {"mean": mean.tolist(), "std": std.tolist(), "objective": cfg.objective}
    return model, stats


def _fetch_bars(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    cfg: AISymbolFilterConfig,
    limit_symbols: int | None,
) -> dict[str, pd.DataFrame]:
    client = StockHistoricalDataClient(api_key, api_secret)
    symbols = symbols[:limit_symbols] if limit_symbols else symbols
    end = datetime.utcnow()
    start = end - timedelta(days=cfg.lookback_days)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for chunk in _chunked(symbols, 200):
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=_map_timeframe(cfg.interval),
            start=start,
            end=end,
            feed=_map_feed(cfg.feed),
        )
        df = _fetch_with_retries(client, request, cfg.timeout_seconds, cfg.retries)
        if df is None or df.empty:
            continue
        if isinstance(df.index, pd.MultiIndex):
            for symbol in chunk:
                try:
                    frame = df.xs(symbol, level="symbol")
                except KeyError:
                    continue
                bars_by_symbol[symbol] = frame.sort_index()
        else:
            for symbol in chunk:
                bars_by_symbol[symbol] = df.sort_index()
    return bars_by_symbol


def _fetch_with_retries(client: StockHistoricalDataClient, request: StockBarsRequest, timeout: int, retries: int):
    for attempt in range(retries + 1):
        try:
            return client.get_stock_bars(request).df
        except Exception as exc:
            if attempt >= retries:
                logging.warning("AI filter bars fetch failed: %s", exc)
                return None
    return None


def _build_training_data(bars: dict[str, pd.DataFrame], cfg: AISymbolFilterConfig):
    features = []
    labels = []
    for frame in bars.values():
        feat, lab = _features_and_labels(frame, cfg)
        if feat.size == 0:
            continue
        if cfg.max_samples_per_symbol and feat.shape[0] > cfg.max_samples_per_symbol:
            feat = feat[-cfg.max_samples_per_symbol :]
            lab = lab[-cfg.max_samples_per_symbol :]
        features.append(feat)
        labels.append(lab)
    if not features:
        return np.empty((0, 0)), np.empty((0,))
    return np.vstack(features), np.concatenate(labels)


def _features_and_labels(frame: pd.DataFrame, cfg: AISymbolFilterConfig):
    if frame is None or frame.empty:
        return np.empty((0, 0)), np.empty((0,))
    close = frame["close"].astype(float).values
    volume = frame["volume"].astype(float).values
    returns = np.diff(close) / close[:-1]
    if len(returns) < cfg.window + 1:
        return np.empty((0, 0)), np.empty((0,))
    feat_rows = []
    labels = []
    for idx in range(cfg.window, len(returns) - 1):
        window_ret = returns[idx - cfg.window : idx]
        window_vol = volume[idx - cfg.window : idx]
        feat_rows.append(_feature_vector(window_ret, window_vol))
        labels.append(_target_value(returns[idx + 1], window_vol, cfg.objective))
    return np.array(feat_rows, dtype=float), np.array(labels, dtype=float)


def _latest_features(frame: pd.DataFrame, window: int):
    if frame is None or frame.empty:
        return None
    close = frame["close"].astype(float).values
    volume = frame["volume"].astype(float).values
    returns = np.diff(close) / close[:-1]
    if len(returns) < window:
        return None
    window_ret = returns[-window:]
    window_vol = volume[-window:]
    return _feature_vector(window_ret, window_vol)


def _feature_vector(returns: np.ndarray, volume: np.ndarray) -> np.ndarray:
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    momentum = float(np.sum(returns))
    last_ret = float(returns[-1])
    vol_mean = float(np.mean(volume)) if volume.size else 0.0
    vol_std = float(np.std(volume)) if volume.size else 0.0
    vol_z = (float(volume[-1]) - vol_mean) / vol_std if vol_std else 0.0
    return np.array([mean_ret, std_ret, momentum, last_ret, vol_z], dtype=float)


def _target_value(next_return: float, volume_window: np.ndarray, objective: str) -> float:
    if objective == "return":
        return float(next_return)
    if objective == "risk_adjusted":
        vol = float(np.std(volume_window)) if volume_window.size else 0.0
        penalty = vol if vol else 1.0
        return float(next_return) / penalty
    if objective == "hit_rate":
        return 1.0 if next_return > 0 else 0.0
    if objective == "liquidity_aware":
        vol_mean = float(np.mean(volume_window)) if volume_window.size else 0.0
        penalty = 1.0 / max(vol_mean, 1.0)
        return float(next_return) - penalty
    if objective == "drawdown_aware":
        penalty = max(-next_return, 0.0)
        return float(next_return) - penalty
    return float(next_return)


def _predict(model, stats: dict, features: np.ndarray) -> float:
    mean = np.array(stats.get("mean", []), dtype=float)
    std = np.array(stats.get("std", []), dtype=float)
    if mean.size and std.size:
        features = (features - mean) / std
    x = torch.tensor(features, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        score = model(x).item()
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return float(score)


def _map_timeframe(interval: str) -> TimeFrame:
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)
    if interval.endswith("h"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
    if interval.endswith("d"):
        return TimeFrame(1, TimeFrameUnit.Day)
    return TimeFrame(1, TimeFrameUnit.Minute)


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _map_feed(feed: str) -> DataFeed:
    if feed.lower() == "sip":
        return DataFeed.SIP
    return DataFeed.IEX

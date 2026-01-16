# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import json
import logging
import math

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from app.data.news import fetch_catalyst_symbols_for_config
from app.data.market_cache import MarketCache, interval_to_seconds
from app.data.yfinance_utils import fetch_yfinance_bars
from app.utils.signal_features import compute_signal_metrics_from_window


@dataclass
class AISymbolFilterConfig:
    interval: str
    lookback_days: int
    window: int
    retrain_hours: int
    model_path: str
    model_type: str
    train_max_symbols: int
    max_samples_per_symbol: int
    online_enabled: bool
    online_learning_rate: float
    online_steps: int
    online_timesteps: int
    online_max_symbols: int
    rl_timesteps: int
    rl_learning_rate: float
    rl_batch_size: int
    rl_n_steps: int
    rl_gamma: float
    rl_ent_coef: float
    rl_clip_range: float
    rl_gae_lambda: float
    news_enabled: bool
    news_lookback_hours: int
    news_keywords: list[str]
    news_timeout_seconds: int
    news_retries: int
    news_base_url: str
    news_provider: str
    timeout_seconds: int
    retries: int
    objective: str
    time_penalty_per_bar: float
    feed: str
    provider: str
    market_cache_enabled: bool
    market_cache_redis_url: str
    market_cache_file_dir: str
    market_cache_cache_only: bool
    market_cache_ignore_staleness: bool


def score_symbols(
    symbols: Iterable[str],
    api_key: str,
    api_secret: str,
    cfg: dict,
    brokers_cfg: dict | None = None,
) -> tuple[list[str], dict[str, float], dict[str, dict[str, float]]]:
    symbols = [s for s in symbols if s]
    if not symbols or not api_key or not api_secret:
        return [], {}, {}
    config = _read_config(cfg)
    logging.info(
        "AI filter config; symbols=%d provider=%s interval=%s lookback_days=%s cache_enabled=%s cache_only=%s cache_ignore_stale=%s",
        len(symbols),
        config.provider,
        config.interval,
        config.lookback_days,
        config.market_cache_enabled,
        config.market_cache_cache_only,
        config.market_cache_ignore_staleness,
    )
    model_path = _normalize_model_path(config.model_path, config.model_type)
    if config.model_type == "ppo":
        model, stats = _load_ppo_model(model_path, config.retrain_hours)
    else:
        model, stats = _load_linear_model(model_path, config.retrain_hours)
    if model is None or stats is None:
        if config.model_type == "ppo":
            model, stats = _train_ppo_model(symbols, api_key, api_secret, config, brokers_cfg or {})
        else:
            model, stats = _train_linear_model(symbols, api_key, api_secret, config, brokers_cfg or {})
        if model is not None and stats is not None:
            if config.model_type == "ppo":
                _save_ppo_model(model_path, model, stats)
            else:
                _save_linear_model(model_path, model, stats)

    if model is None or stats is None:
        return symbols, {s: 0.0 for s in symbols}

    catalyst_map = _fetch_news_catalysts(symbols, api_key, api_secret, config, brokers_cfg or {})
    if config.online_enabled:
        if config.model_type == "ppo":
            updated = _online_update_ppo_model(symbols, api_key, api_secret, config, model, stats, catalyst_map)
        else:
            updated = _online_update_linear_model(symbols, api_key, api_secret, config, model, stats, catalyst_map)
        if updated:
            if config.model_type == "ppo":
                _save_ppo_model(model_path, model, stats)
            else:
                _save_linear_model(model_path, model, stats)

    bars = _fetch_bars(symbols, api_key, api_secret, config, limit_symbols=None)
    scores = {}
    signal_map: dict[str, dict[str, float]] = {}
    for symbol, frame in bars.items():
        features = _latest_features(frame, config.window, catalyst_map.get(symbol, False), config.interval)
        if features is None:
            scores[symbol] = 0.0
            continue
        if config.model_type == "ppo":
            score = _predict_ppo(model, stats, features)
        else:
            score = _predict_linear(model, stats, features)
        scores[symbol] = float(score)
        if frame is not None and not frame.empty:
            try:
                idx = frame.index
                last_day = idx[-1].date()
                day_frame = frame.loc[idx.date == last_day]
                if day_frame.empty:
                    day_frame = frame
                signal_map[symbol] = compute_signal_metrics_from_window(
                    prices=day_frame["close"].astype(float).tolist(),
                    volumes=day_frame["volume"].astype(float).tolist(),
                    highs=day_frame["high"].astype(float).tolist() if "high" in day_frame else None,
                    lows=day_frame["low"].astype(float).tolist() if "low" in day_frame else None,
                    interval=config.interval,
                )
            except Exception:
                signal_map[symbol] = {}
    for symbol in symbols:
        scores.setdefault(symbol, 0.0)
        signal_map.setdefault(symbol, {})
    ordered = sorted(symbols, key=lambda s: scores.get(s, 0.0), reverse=True)
    return ordered, scores, signal_map


def _read_config(cfg: dict) -> AISymbolFilterConfig:
    interval = str(cfg.get("interval", "5m"))
    lookback_days = int(cfg.get("lookback_days", 20))
    window = int(cfg.get("window", 20))
    retrain_hours = int(cfg.get("retrain_hours", 6))
    model_path = str(cfg.get("model_path", "/data/ai_symbol_filter.zip"))
    model_type = str(cfg.get("model_type", "ppo")).lower()
    train_max_symbols = int(cfg.get("train_max_symbols", 300))
    max_samples_per_symbol = int(cfg.get("max_samples_per_symbol", 200))
    rl_cfg = cfg.get("rl", {}) or {}
    rl_timesteps = int(rl_cfg.get("timesteps", 20000))
    rl_learning_rate = float(rl_cfg.get("learning_rate", 0.0003))
    rl_batch_size = int(rl_cfg.get("batch_size", 64))
    rl_n_steps = int(rl_cfg.get("n_steps", 256))
    rl_gamma = float(rl_cfg.get("gamma", 0.99))
    rl_ent_coef = float(rl_cfg.get("ent_coef", 0.01))
    rl_clip_range = float(rl_cfg.get("clip_range", 0.2))
    rl_gae_lambda = float(rl_cfg.get("gae_lambda", 0.95))
    online_cfg = cfg.get("online", {}) or {}
    online_enabled = bool(online_cfg.get("enabled", False))
    online_learning_rate = float(online_cfg.get("learning_rate", 0.001))
    online_steps = int(online_cfg.get("steps", 5))
    online_timesteps = int(online_cfg.get("timesteps", online_steps))
    online_max_symbols = int(online_cfg.get("max_symbols", 200))
    news_cfg = cfg.get("news", {}) or {}
    news_enabled = bool(news_cfg.get("enabled", False))
    news_lookback_hours = int(news_cfg.get("lookback_hours", 12))
    news_keywords = list(news_cfg.get("keywords", []))
    news_timeout_seconds = int(news_cfg.get("timeout_seconds", 10))
    news_retries = int(news_cfg.get("retries", 2))
    news_base_url = str(news_cfg.get("base_url", "https://data.alpaca.markets"))
    news_provider = str(news_cfg.get("provider", "alpaca"))
    timeout_seconds = int(cfg.get("timeout_seconds", 15))
    retries = int(cfg.get("retries", 2))
    objective = str(cfg.get("objective", "return"))
    time_penalty_per_bar = float(cfg.get("time_penalty_per_bar", 0.0))
    feed = str(cfg.get("feed", "iex"))
    provider = str(cfg.get("provider", "alpaca"))
    cache_cfg = cfg.get("market_cache", {}) if isinstance(cfg, dict) else {}
    market_cache_enabled = bool(cache_cfg.get("enabled", False))
    market_cache_redis_url = str(cache_cfg.get("redis_url", "redis://redis:6379/0"))
    market_cache_file_dir = str(cache_cfg.get("file_dir", "/data/market_cache"))
    market_cache_cache_only = bool(cache_cfg.get("cache_only", True))
    market_cache_ignore_staleness = bool(cache_cfg.get("ignore_staleness", False))
    return AISymbolFilterConfig(
        interval=interval,
        lookback_days=lookback_days,
        window=window,
        retrain_hours=retrain_hours,
        model_path=model_path,
        model_type=model_type,
        train_max_symbols=train_max_symbols,
        max_samples_per_symbol=max_samples_per_symbol,
        online_enabled=online_enabled,
        online_learning_rate=online_learning_rate,
        online_steps=online_steps,
        online_timesteps=online_timesteps,
        online_max_symbols=online_max_symbols,
        rl_timesteps=rl_timesteps,
        rl_learning_rate=rl_learning_rate,
        rl_batch_size=rl_batch_size,
        rl_n_steps=rl_n_steps,
        rl_gamma=rl_gamma,
        rl_ent_coef=rl_ent_coef,
        rl_clip_range=rl_clip_range,
        rl_gae_lambda=rl_gae_lambda,
        news_enabled=news_enabled,
        news_lookback_hours=news_lookback_hours,
        news_keywords=news_keywords,
        news_timeout_seconds=news_timeout_seconds,
        news_retries=news_retries,
        news_base_url=news_base_url,
        news_provider=news_provider,
        timeout_seconds=timeout_seconds,
        retries=retries,
        objective=objective,
        time_penalty_per_bar=time_penalty_per_bar,
        feed=feed,
        provider=provider,
        market_cache_enabled=market_cache_enabled,
        market_cache_redis_url=market_cache_redis_url,
        market_cache_file_dir=market_cache_file_dir,
        market_cache_cache_only=market_cache_cache_only,
        market_cache_ignore_staleness=market_cache_ignore_staleness,
    )


def _normalize_model_path(model_path: str, model_type: str) -> Path:
    path = Path(model_path)
    if model_type == "ppo" and path.suffix == ".pt":
        return path.with_suffix(".zip")
    return path


def _meta_path(model_path: Path) -> Path:
    suffix = model_path.suffix if model_path.suffix else ".zip"
    return model_path.with_suffix(f"{suffix}.meta.json")


def _load_linear_model(model_path: Path, retrain_hours: int):
    if not model_path.exists():
        return None, None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("AI filter load device: %s", device)
        payload = torch.load(model_path, map_location=device)
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
    model.to(device)
    model.eval()
    stats = {
        "mean": payload.get("mean"),
        "std": payload.get("std"),
        "objective": payload.get("objective"),
    }
    return model, stats


def _save_linear_model(model_path: Path, model, stats: dict):
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "input_dim": model.in_features,
        "state_dict": state_dict,
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "objective": stats.get("objective"),
        "trained_at": datetime.utcnow().isoformat(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, model_path)


def _train_linear_model(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    cfg: AISymbolFilterConfig,
    brokers_cfg: dict,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("AI filter train device: %s", device)
    train_symbols = symbols[: cfg.train_max_symbols]
    bars = _fetch_bars(train_symbols, api_key, api_secret, cfg, limit_symbols=cfg.train_max_symbols)
    catalyst_map = _fetch_news_catalysts(train_symbols, api_key, api_secret, cfg, brokers_cfg or {})
    features, labels = _build_training_data(bars, cfg, catalyst_map)
    if features.size == 0:
        logging.warning("AI filter training skipped: no data")
        return None, None
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    features = (features - mean) / std
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)
    model = torch.nn.Linear(x.shape[1], 1).to(device)
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


def _online_update_linear_model(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    cfg: AISymbolFilterConfig,
    model,
    stats: dict,
    catalyst_map: dict[str, bool],
) -> bool:
    device = next(model.parameters()).device
    update_symbols = symbols[: cfg.online_max_symbols]
    if not update_symbols:
        return False
    bars = _fetch_bars(update_symbols, api_key, api_secret, cfg, limit_symbols=cfg.online_max_symbols)
    features, labels = _build_training_data(bars, cfg, catalyst_map)
    if features.size == 0:
        logging.warning("AI filter online update skipped: no data")
        return False
    mean = np.array(stats.get("mean") or [], dtype=float)
    std = np.array(stats.get("std") or [], dtype=float)
    if mean.size and std.size:
        std = np.where(std == 0, 1.0, std)
        features = (features - mean) / std
    else:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        features = (features - mean) / std
        stats["mean"] = mean.tolist()
        stats["std"] = std.tolist()
    stats.setdefault("objective", cfg.objective)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.online_learning_rate)
    loss_fn = torch.nn.MSELoss()
    model.train()
    last_loss = None
    for _ in range(cfg.online_steps):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    model.eval()
    if last_loss is not None:
        logging.info("AI filter online update complete; loss=%.6f samples=%d", last_loss, len(labels))
    return True


class _SymbolFilterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, rewards: np.ndarray, shuffle: bool = True):
        super().__init__()
        self._features = np.asarray(features, dtype=np.float32)
        self._rewards = np.asarray(rewards, dtype=np.float32)
        self._shuffle = shuffle
        self._order = np.arange(len(self._features))
        self._idx = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._features.shape[1],),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._shuffle and len(self._order) > 1:
            np.random.shuffle(self._order)
        self._idx = 0
        obs = self._features[self._order[self._idx]]
        return obs, {}

    def step(self, action: int):
        reward = float(self._rewards[self._order[self._idx]]) if int(action) == 1 else 0.0
        self._idx += 1
        done = self._idx >= len(self._order)
        if done:
            obs = self._features[self._order[-1]]
        else:
            obs = self._features[self._order[self._idx]]
        return obs, reward, done, False, {}


def _build_env(features: np.ndarray, rewards: np.ndarray) -> DummyVecEnv:
    return DummyVecEnv([lambda: _SymbolFilterEnv(features, rewards, shuffle=True)])


def _load_ppo_model(model_path: Path, retrain_hours: int):
    if not model_path.exists():
        return None, None
    meta_path = _meta_path(model_path)
    if not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        logging.warning("AI filter metadata load failed: %s", exc)
        return None, None
    trained_at = meta.get("trained_at")
    if trained_at:
        trained_dt = datetime.fromisoformat(trained_at)
        if datetime.utcnow() - trained_dt > timedelta(hours=retrain_hours):
            return None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("AI filter load device: %s", device)
    try:
        model = PPO.load(str(model_path), device=device)
    except Exception as exc:
        logging.warning("AI filter PPO load failed: %s", exc)
        return None, None
    return model, meta


def _save_ppo_model(model_path: Path, model: PPO, stats: dict):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    meta = {
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "objective": stats.get("objective"),
        "trained_at": datetime.utcnow().isoformat(),
        "model_type": "ppo",
    }
    _meta_path(model_path).write_text(json.dumps(meta, indent=2, sort_keys=True))


def _train_ppo_model(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    cfg: AISymbolFilterConfig,
    brokers_cfg: dict,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("AI filter train device: %s", device)
    train_symbols = symbols[: cfg.train_max_symbols]
    logging.info("AI filter training start; symbols=%d timesteps=%d", len(train_symbols), cfg.rl_timesteps)
    bars = _fetch_bars(train_symbols, api_key, api_secret, cfg, limit_symbols=cfg.train_max_symbols)
    catalyst_map = _fetch_news_catalysts(train_symbols, api_key, api_secret, cfg, brokers_cfg or {})
    features, labels = _build_training_data(bars, cfg, catalyst_map)
    if features.size == 0:
        logging.warning("AI filter training skipped: no data")
        return None, None
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    features = (features - mean) / std
    env = _build_env(features, labels)
    timesteps = max(int(cfg.rl_timesteps), int(cfg.rl_n_steps))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.rl_learning_rate,
        n_steps=max(16, int(cfg.rl_n_steps)),
        batch_size=max(16, int(cfg.rl_batch_size)),
        gamma=cfg.rl_gamma,
        ent_coef=cfg.rl_ent_coef,
        clip_range=cfg.rl_clip_range,
        gae_lambda=cfg.rl_gae_lambda,
        verbose=0,
        device=device,
    )
    model.learn(total_timesteps=timesteps)
    stats = {"mean": mean.tolist(), "std": std.tolist(), "objective": cfg.objective}
    logging.info(
        "AI filter training complete; symbols=%d samples=%d timesteps=%d",
        len(train_symbols),
        len(labels),
        timesteps,
    )
    return model, stats


def _online_update_ppo_model(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    cfg: AISymbolFilterConfig,
    model: PPO,
    stats: dict,
    catalyst_map: dict[str, bool],
) -> bool:
    update_symbols = symbols[: cfg.online_max_symbols]
    if not update_symbols:
        return False
    logging.info("AI filter online update start; symbols=%d timesteps=%d", len(update_symbols), cfg.online_timesteps)
    bars = _fetch_bars(update_symbols, api_key, api_secret, cfg, limit_symbols=cfg.online_max_symbols)
    features, labels = _build_training_data(bars, cfg, catalyst_map)
    if features.size == 0:
        logging.warning("AI filter online update skipped: no data")
        return False
    mean = np.array(stats.get("mean") or [], dtype=float)
    std = np.array(stats.get("std") or [], dtype=float)
    if mean.size and std.size:
        std = np.where(std == 0, 1.0, std)
        features = (features - mean) / std
    else:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        features = (features - mean) / std
        stats["mean"] = mean.tolist()
        stats["std"] = std.tolist()
    stats.setdefault("objective", cfg.objective)
    env = _build_env(features, labels)
    model.set_env(env)
    timesteps = max(int(cfg.online_timesteps), int(cfg.rl_n_steps))
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
    logging.info("AI filter online update complete; timesteps=%d samples=%d", timesteps, len(labels))
    return True


def _fetch_bars(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    cfg: AISymbolFilterConfig,
    limit_symbols: int | None,
) -> dict[str, pd.DataFrame]:
    if cfg.provider.lower() == "yfinance":
        return _fetch_bars_yfinance(symbols, cfg, limit_symbols)
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
            if len(chunk) == 1:
                bars_by_symbol[chunk[0]] = df.sort_index()
            elif "symbol" in df.columns:
                for symbol, frame in df.groupby("symbol"):
                    bars_by_symbol[str(symbol)] = frame.drop(columns=["symbol"]).sort_index()
            else:
                logging.warning("AI filter bars missing symbol index for %d symbols; skipping chunk.", len(chunk))
    return bars_by_symbol


def _fetch_bars_yfinance(
    symbols: list[str],
    cfg: AISymbolFilterConfig,
    limit_symbols: int | None,
) -> dict[str, pd.DataFrame]:
    symbols = symbols[:limit_symbols] if limit_symbols else symbols
    max_age = interval_to_seconds(cfg.interval)
    if cfg.market_cache_enabled:
        cache = MarketCache(
            cfg.market_cache_redis_url,
            cfg.market_cache_file_dir,
            ignore_staleness=cfg.market_cache_ignore_staleness,
        )
        cached = cache.get_bars(symbols, cfg.interval, max_age_seconds=max_age, lowercase=True)
        if cfg.market_cache_cache_only:
            required = {"close", "volume"}
            logging.info(
                "AI filter cache-only bars; cached=%d missing=%d",
                len(cached),
                max(0, len(symbols) - len(cached)),
            )
            return {symbol: frame for symbol, frame in cached.items() if required.issubset(frame.columns)}
        missing = [symbol for symbol in symbols if symbol not in cached]
    else:
        cached = {}
        missing = symbols
    if missing:
        fetched = fetch_yfinance_bars(
            missing,
            cfg.lookback_days,
            cfg.interval,
            batch_size=100,
            lowercase=True,
            drop_zero_volume=False,
        )
        if cfg.market_cache_enabled and fetched:
            cache.set_bars(fetched, cfg.interval, ttl_seconds=max_age)
        cached.update(fetched)
        logging.info(
            "AI filter fetched bars; fetched=%d missing_after=%d",
            len(fetched),
            max(0, len(symbols) - len(cached)),
        )
    required = {"close", "volume"}
    return {symbol: frame for symbol, frame in cached.items() if required.issubset(frame.columns)}


def _fetch_with_retries(client: StockHistoricalDataClient, request: StockBarsRequest, timeout: int, retries: int):
    for attempt in range(retries + 1):
        try:
            return client.get_stock_bars(request).df
        except Exception as exc:
            if attempt >= retries:
                logging.warning("AI filter bars fetch failed: %s", exc)
                return None
    return None


def _build_training_data(bars: dict[str, pd.DataFrame], cfg: AISymbolFilterConfig, catalyst_map: dict[str, bool]):
    features = []
    labels = []
    for symbol, frame in bars.items():
        catalyst = catalyst_map.get(symbol, False)
        feat, lab = _features_and_labels(frame, cfg, catalyst)
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


def _features_and_labels(frame: pd.DataFrame, cfg: AISymbolFilterConfig, catalyst: bool):
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
        window_prices = close[idx - cfg.window : idx + 1]
        window_vol_prices = volume[idx - cfg.window : idx + 1]
        feat_rows.append(
            _feature_vector(window_ret, window_vol, catalyst, window_prices, window_vol_prices, cfg.interval)
        )
        labels.append(_target_value(returns[idx + 1], window_vol, cfg.objective, cfg.time_penalty_per_bar))
    return np.array(feat_rows, dtype=float), np.array(labels, dtype=float)


def _latest_features(frame: pd.DataFrame, window: int, catalyst: bool, interval: str):
    if frame is None or frame.empty:
        return None
    close = frame["close"].astype(float).values
    volume = frame["volume"].astype(float).values
    returns = np.diff(close) / close[:-1]
    if len(returns) < window:
        return None
    window_ret = returns[-window:]
    window_vol = volume[-window:]
    window_prices = close[-(window + 1) :]
    window_vol_prices = volume[-(window + 1) :]
    return _feature_vector(window_ret, window_vol, catalyst, window_prices, window_vol_prices, interval)


def build_feature_vector_from_series(
    prices: Iterable[float],
    volumes: Iterable[float],
    window: int,
    catalyst: bool,
    interval: str,
) -> np.ndarray | None:
    close = np.array(list(prices), dtype=float)
    if close.size < 2:
        return None
    volume = np.array(list(volumes), dtype=float)
    returns = np.diff(close) / close[:-1]
    if len(returns) < window:
        return None
    window_ret = returns[-window:]
    if volume.size >= window:
        window_vol = volume[-window:]
    else:
        window_vol = np.zeros(window, dtype=float)
        if volume.size:
            window_vol[-volume.size :] = volume
    window_prices = close[-(window + 1) :]
    if volume.size >= window + 1:
        window_vol_prices = volume[-(window + 1) :]
    else:
        window_vol_prices = volume
    return _feature_vector(window_ret, window_vol, catalyst, window_prices, window_vol_prices, interval)


def latest_features_for_symbol(
    symbol: str,
    api_key: str,
    api_secret: str,
    cfg: dict | AISymbolFilterConfig,
    catalyst: bool,
) -> np.ndarray | None:
    if not symbol or not api_key or not api_secret:
        return None
    config = cfg if isinstance(cfg, AISymbolFilterConfig) else _read_config(cfg)
    bars = _fetch_bars([symbol], api_key, api_secret, config, limit_symbols=None)
    if not bars or symbol not in bars:
        return None
    return _latest_features(bars.get(symbol), config.window, catalyst, config.interval)


def _feature_vector(
    returns: np.ndarray,
    volume: np.ndarray,
    catalyst: bool,
    prices: np.ndarray | None,
    prices_volume: np.ndarray | None,
    interval: str,
) -> np.ndarray:
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    momentum = float(np.sum(returns))
    last_ret = float(returns[-1])
    vol_mean = float(np.mean(volume)) if volume.size else 0.0
    vol_std = float(np.std(volume)) if volume.size else 0.0
    vol_z = (float(volume[-1]) - vol_mean) / vol_std if vol_std else 0.0
    catalyst_flag = 1.0 if catalyst else 0.0
    features = [mean_ret, std_ret, momentum, last_ret, vol_z, catalyst_flag]
    if prices is not None and prices.size > 1:
        vol_series = prices_volume if prices_volume is not None and prices_volume.size > 0 else volume
        signal_vals = compute_signal_metrics_from_window(
            prices=prices.tolist(),
            volumes=vol_series.tolist(),
            interval=interval,
        )
        features.extend(
            [
                signal_vals.get("signal_30m_return_pct", 0.0),
                signal_vals.get("signal_60m_return_pct", 0.0),
                signal_vals.get("signal_early_volume_pct", 0.0),
                signal_vals.get("signal_runup_pct", 0.0),
                signal_vals.get("signal_drawdown_pct", 0.0),
                signal_vals.get("signal_abs_move", 0.0),
                signal_vals.get("signal_runup_abs", 0.0),
                signal_vals.get("signal_drawdown_abs", 0.0),
            ]
        )
    return np.array(features, dtype=float)


def _target_value(
    next_return: float,
    volume_window: np.ndarray,
    objective: str,
    time_penalty_per_bar: float,
) -> float:
    if objective == "return":
        return float(next_return)
    if objective == "return_time_penalty":
        return float(next_return) - time_penalty_per_bar
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


def _predict_linear(model, stats: dict, features: np.ndarray) -> float:
    mean = np.array(stats.get("mean", []), dtype=float)
    std = np.array(stats.get("std", []), dtype=float)
    if mean.size and std.size:
        features = (features - mean) / std
    x = torch.tensor(features, dtype=torch.float32, device=next(model.parameters()).device).view(1, -1)
    with torch.no_grad():
        score = model(x).item()
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return float(score)


def _predict_ppo(model: PPO, stats: dict, features: np.ndarray) -> float:
    mean = np.array(stats.get("mean", []), dtype=float)
    std = np.array(stats.get("std", []), dtype=float)
    if mean.size and std.size:
        features = (features - mean) / std
    obs = np.array(features, dtype=np.float32).reshape(1, -1)
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    dist = model.policy.get_distribution(obs_tensor)
    probs = dist.distribution.probs.detach().cpu().numpy()
    if probs.ndim == 2 and probs.shape[1] >= 2:
        score = float(probs[0][1])
    else:
        score = float(probs.squeeze()[()])
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return score


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


def _fetch_news_catalysts(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    cfg: AISymbolFilterConfig,
    brokers_cfg: dict,
) -> dict[str, bool]:
    if not cfg.news_enabled:
        return {}
    news_cfg = {
        "provider": cfg.news_provider,
        "base_url": cfg.news_base_url,
        "api_key": api_key,
        "api_secret": api_secret,
        "lookback_hours": cfg.news_lookback_hours,
        "keywords": cfg.news_keywords,
        "timeout_seconds": cfg.news_timeout_seconds,
        "retries": cfg.news_retries,
    }
    return fetch_catalyst_symbols_for_config(symbols, news_cfg, brokers_cfg)

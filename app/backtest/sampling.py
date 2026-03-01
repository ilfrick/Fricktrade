# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd


@dataclass(frozen=True)
class BacktestWindow:
    start: datetime
    end: datetime
    symbols: list[str]


def _load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df["Datetime"] = df["Datetime"].dt.tz_convert(None)
    df = df.set_index("Datetime").sort_index()
    return df


def _dollar_volume(frame: pd.DataFrame) -> float:
    if "Close" not in frame or "Volume" not in frame:
        return 0.0
    dv = frame["Close"] * frame["Volume"]
    if dv.empty:
        return 0.0
    return float(dv.median())


def _symbols_from_data_dir(data_dir: Path, interval: str) -> list[str]:
    pattern = f"*_{interval}.csv"
    symbols = []
    for path in data_dir.glob(pattern):
        name = path.stem
        suffix = f"_{interval}"
        if not name.endswith(suffix):
            continue
        symbol = name[: -len(suffix)]
        if symbol:
            symbols.append(symbol.replace("_", "."))
    return sorted(set(symbols))


def _build_tiers(symbols: list[str], data_dir: Path, interval: str, tiers: int) -> list[list[str]]:
    if tiers <= 1:
        return [symbols]
    stats = []
    for symbol in symbols:
        path = data_dir / f"{symbol.replace('.', '_')}_{interval}.csv"
        if not path.exists():
            continue
        frame = _load_frame(path)
        stats.append((symbol, _dollar_volume(frame)))
    stats.sort(key=lambda item: item[1], reverse=True)
    if not stats:
        return [symbols]
    tier_size = max(1, int(len(stats) / tiers))
    buckets = []
    for idx in range(tiers):
        start = idx * tier_size
        end = None if idx == tiers - 1 else (idx + 1) * tier_size
        bucket = [symbol for symbol, _ in stats[start:end]]
        if bucket:
            buckets.append(bucket)
    return buckets or [symbols]


def sample_universe(
    data_dir: Path,
    interval: str,
    sample_per_tier: int,
    tiers: int,
    seed: int,
) -> list[str]:
    symbols = _symbols_from_data_dir(data_dir, interval)
    if not symbols:
        return []
    rng = random.Random(seed)
    buckets = _build_tiers(symbols, data_dir, interval, tiers)
    selected: list[str] = []
    for bucket in buckets:
        if not bucket:
            continue
        if sample_per_tier <= 0 or sample_per_tier >= len(bucket):
            selected.extend(bucket)
        else:
            selected.extend(rng.sample(bucket, sample_per_tier))
    return sorted(set(selected))


def build_windows(
    start: datetime,
    end: datetime,
    window_days: int,
    step_days: int,
) -> list[tuple[datetime, datetime]]:
    if window_days <= 0:
        return []
    step_days = max(1, step_days)
    windows = []
    cursor = start
    while cursor <= end:
        win_end = min(end, cursor + timedelta(days=window_days))
        windows.append((cursor, win_end))
        cursor = cursor + timedelta(days=step_days)
        if win_end == end:
            break
    return windows


def build_walkforward_windows(
    start: datetime,
    end: datetime,
    train_days: int = 180,
    embargo_days: int = 5,
    test_days: int = 30,
    step_days: int = 30,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Return list of (train_start, train_end, test_start, test_end) walk-forward folds.

    Each fold has an embargo gap between train_end and test_start to prevent
    lookahead leakage from recent training samples.
    """
    if train_days <= 0 or test_days <= 0:
        return []
    step_days = max(1, step_days)
    folds = []
    # First test window starts after the first training period + embargo
    test_start = start + timedelta(days=train_days + embargo_days)
    while test_start <= end:
        test_end = min(end, test_start + timedelta(days=test_days))
        train_end = test_start - timedelta(days=embargo_days)
        train_start = train_end - timedelta(days=train_days)
        if train_start < start:
            train_start = start
        folds.append((train_start, train_end, test_start, test_end))
        if test_end >= end:
            break
        test_start = test_start + timedelta(days=step_days)
    return folds


def build_backtest_plan(cfg: dict) -> list[BacktestWindow]:
    backtest_cfg = cfg.get("backtest", {})
    plan_cfg = backtest_cfg.get("plan", {})
    if not plan_cfg.get("enabled", False):
        return []
    data_cfg = cfg.get("data", {})
    interval = str(data_cfg.get("interval", "1m"))
    data_dir = Path(backtest_cfg.get("data_dir", "/data"))
    sample_per_tier = int(plan_cfg.get("sample_per_tier", 0))
    tiers = int(plan_cfg.get("liquidity_tiers", 1))
    seed = int(plan_cfg.get("seed", 42))
    symbols = sample_universe(data_dir, interval, sample_per_tier, tiers, seed)
    if not symbols:
        return []
    start = datetime.strptime(backtest_cfg["start"], "%Y-%m-%d")
    end = datetime.strptime(backtest_cfg["end"], "%Y-%m-%d")
    window_days = int(plan_cfg.get("window_days", 60))
    step_days = int(plan_cfg.get("step_days", 30))
    windows = build_windows(start, end, window_days, step_days)
    return [BacktestWindow(start=win_start, end=win_end, symbols=symbols) for win_start, win_end in windows]

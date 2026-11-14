# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Calendar-based, common-timestamp validation splits with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class CalendarSplit:
    train: dict[str, pd.DataFrame]
    holdout: dict[str, pd.DataFrame]
    cutoff: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    common_index: pd.DatetimeIndex


def common_timestamp_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict all symbols to their shared timestamps and return sorted copies."""
    if not frames:
        return {}
    indexes = [pd.DatetimeIndex(pd.to_datetime(f.index)).drop_duplicates().sort_values()
               for f in frames.values()]
    common = indexes[0]
    for idx in indexes[1:]:
        common = common.intersection(idx)
    common = common.sort_values()
    return {symbol: frame.loc[frame.index.intersection(common)].sort_index()
            for symbol, frame in frames.items()}


def calendar_holdout_split(
    frames: dict[str, pd.DataFrame], fraction: float = 0.75,
    holding_period: timedelta = timedelta(0),
) -> CalendarSplit:
    """Use one timestamp cut across symbols, purge crossing labels, embargo start.

    For trade-like rows with `entry_time` and `exit_time`, train observations opened
    before the cut but closed on/after it are purged. The holding-period embargo is
    applied to the beginning of holdout. Raw OHLCV frames are split at the same
    calendar timestamp after intersecting their indexes.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    aligned = common_timestamp_frames(frames)
    if not aligned or not next(iter(aligned.values())).shape[0]:
        raise ValueError("no common timestamps across dataset symbols")
    index = next(iter(aligned.values())).index
    cut_pos = min(max(int(len(index) * fraction), 1), len(index) - 1)
    cutoff = pd.Timestamp(index[cut_pos])
    holdout_start = cutoff + holding_period
    train, holdout = {}, {}
    for symbol, frame in aligned.items():
        train[symbol] = frame.loc[frame.index < cutoff]
        holdout[symbol] = frame.loc[frame.index >= holdout_start]
    return CalendarSplit(
        train=train, holdout=holdout, cutoff=cutoff,
        holdout_start=holdout_start,
        holdout_end=pd.Timestamp(index[-1]),
        common_index=index,
    )


def purge_crossing_trades(
    trades: pd.DataFrame, cutoff: pd.Timestamp, embargo: timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/holdout trades, removing any trade that crosses the cutoff."""
    if not {"entry_time", "exit_time"}.issubset(trades.columns):
        raise ValueError("trades must contain entry_time and exit_time")
    entry = pd.to_datetime(trades["entry_time"], utc=True)
    exit_ = pd.to_datetime(trades["exit_time"], utc=True)
    cut = pd.Timestamp(cutoff)
    cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
    hold_start = cut + embargo
    train_mask = (entry < cut) & (exit_ < cut)
    hold_mask = (entry >= hold_start) & (exit_ >= hold_start)
    return trades.loc[train_mask].copy(), trades.loc[hold_mask].copy()


def block_bootstrap_ci(
    values: list[float] | tuple[float, ...],
    block_length: int,
    *,
    iterations: int = 4000,
    confidence: float = 0.95,
    seed: int = 1729,
    trim_fraction: float = 0.01,
) -> dict[str, float]:
    """Circular moving-block bootstrap CIs for mean and upper-trimmed mean."""
    import numpy as np

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"mean_low": float("nan"), "mean_high": float("nan"),
                "trimmed_low": float("nan"), "trimmed_high": float("nan")}
    block = min(max(int(block_length), 1), data.size)
    n_blocks = (data.size + block - 1) // block
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    means, trimmed = np.empty(iterations), np.empty(iterations)
    for i in range(iterations):
        starts = rng.integers(0, data.size, size=n_blocks)
        sample = data[(starts[:, None] + offsets) % data.size].ravel()[:data.size]
        means[i] = sample.mean()
        keep = max(int(np.ceil(sample.size * (1.0 - trim_fraction))), 1)
        trimmed[i] = np.sort(sample)[:keep].mean()
    alpha = (1.0 - confidence) / 2.0
    q = [alpha, 1.0 - alpha]
    return {
        "mean_low": float(np.quantile(means, q[0])),
        "mean_high": float(np.quantile(means, q[1])),
        "trimmed_low": float(np.quantile(trimmed, q[0])),
        "trimmed_high": float(np.quantile(trimmed, q[1])),
    }

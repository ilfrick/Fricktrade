# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from typing import Iterable

import pandas as pd


def interval_minutes(interval: str) -> int:
    if not interval:
        return 1
    if interval.endswith("m"):
        return max(int(interval[:-1]), 1)
    if interval.endswith("h"):
        return max(int(interval[:-1]) * 60, 1)
    if interval.endswith("d"):
        return 24 * 60
    return 1


def compute_signal_metrics_from_df(data: pd.DataFrame, interval: str) -> dict[str, float]:
    if data is None or data.empty:
        return {}
    idx = data.index
    if not isinstance(idx, pd.DatetimeIndex):
        return {}
    if isinstance(idx, pd.MultiIndex):
        data = data.copy()
        data.index = data.index.get_level_values(-1)
        idx = data.index
        if not isinstance(idx, pd.DatetimeIndex):
            return {}
    last_day = idx[-1].date()
    day_mask = idx.date == last_day
    day_data = data.loc[day_mask]
    if day_data.empty:
        return {}
    return compute_signal_metrics_from_window(
        prices=_series_values(day_data, "Close"),
        volumes=_series_values(day_data, "Volume"),
        highs=_series_values(day_data, "High"),
        lows=_series_values(day_data, "Low"),
        interval=interval,
    )


def compute_signal_metrics_from_window(
    prices: Iterable[float],
    volumes: Iterable[float],
    interval: str,
    highs: Iterable[float] | None = None,
    lows: Iterable[float] | None = None,
) -> dict[str, float]:
    price_list = [float(p) for p in prices if p is not None]
    if len(price_list) < 2:
        return {}
    volume_list = [float(v) for v in volumes if v is not None]
    high_list = [float(v) for v in highs if v is not None] if highs is not None else price_list
    low_list = [float(v) for v in lows if v is not None] if lows is not None else price_list

    open_px = price_list[0]
    close_px = price_list[-1]
    if open_px == 0:
        return {}
    high_px = max(high_list) if high_list else close_px
    low_px = min(low_list) if low_list else close_px

    minutes = interval_minutes(interval)
    bars_30 = max(int(30 / minutes), 1)
    bars_60 = max(int(60 / minutes), 1)

    idx_30 = min(bars_30, len(price_list)) - 1
    idx_60 = min(bars_60, len(price_list)) - 1

    return_30m = (price_list[idx_30] - open_px) / open_px * 100.0
    return_60m = (price_list[idx_60] - open_px) / open_px * 100.0

    total_vol = float(sum(volume_list)) if volume_list else 0.0
    early_vol = float(sum(volume_list[:idx_30 + 1])) if volume_list else 0.0
    early_vol_pct = (early_vol / total_vol * 100.0) if total_vol else 0.0

    runup_pct = (high_px - open_px) / open_px * 100.0
    drawdown_pct = (low_px - open_px) / open_px * 100.0

    return {
        "signal_30m_return_pct": return_30m,
        "signal_60m_return_pct": return_60m,
        "signal_early_volume_pct": early_vol_pct,
        "signal_runup_pct": runup_pct,
        "signal_drawdown_pct": drawdown_pct,
        "signal_abs_move": close_px - open_px,
        "signal_runup_abs": high_px - open_px,
        "signal_drawdown_abs": low_px - open_px,
    }


def _series_values(frame: pd.DataFrame, column: str) -> list[float]:
    if column not in frame.columns:
        return []
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.astype(float).tolist()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


@dataclass
class AlgoSlice:
    qty: int
    earliest_at: datetime


def twap_slices(total_qty: int, duration_seconds: int, slice_count: int) -> list[AlgoSlice]:
    if total_qty <= 0 or slice_count <= 0:
        return []
    slice_count = min(slice_count, total_qty)
    now = datetime.now(timezone.utc)
    qty_per = total_qty // slice_count
    remainder = total_qty % slice_count
    slices: list[AlgoSlice] = []
    for idx in range(slice_count):
        qty = qty_per + (1 if idx < remainder else 0)
        delay = int(duration_seconds * idx / max(slice_count - 1, 1))
        slices.append(AlgoSlice(qty=qty, earliest_at=now + timedelta(seconds=delay)))
    return slices


def pov_slices(total_qty: int, max_participation: float, est_volume: float) -> list[AlgoSlice]:
    if total_qty <= 0:
        return []
    allowed = max(1, int(est_volume * max_participation))
    slices = []
    remaining = total_qty
    now = datetime.now(timezone.utc)
    while remaining > 0:
        qty = min(allowed, remaining)
        slices.append(AlgoSlice(qty=qty, earliest_at=now))
        remaining -= qty
    return slices


def adaptive_slices(
    total_qty: int,
    duration_seconds: int,
    base_slices: int,
    regime: str = "medium_vol_normal",
    realized_vol: float = 0.0,
) -> list[AlgoSlice]:
    """Regime-aware TWAP: more/shorter slices in crisis, fewer in calm."""
    if regime == "high_vol_crisis":
        slices = min(base_slices * 2, 16)
        dur = max(duration_seconds // 2, 30)
    elif regime == "low_vol_trending" and realized_vol < 0.005:
        slices = max(base_slices - 1, 2)
        dur = int(duration_seconds * 1.25)
    else:
        slices = base_slices
        dur = duration_seconds
    return twap_slices(total_qty, dur, slices)


def vwap_slices(total_qty: int, volume_profile: Iterable[float], duration_seconds: int) -> list[AlgoSlice]:
    profile = list(volume_profile)
    if total_qty <= 0 or not profile:
        return []
    slice_count = min(len(profile), total_qty)
    profile = profile[:slice_count]
    total = sum(profile) or 1.0
    weights = list(profile)
    quantities = [int(total_qty * (w / total)) for w in weights]
    remainder = total_qty - sum(quantities)
    if remainder > 0:
        order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        for idx in order:
            if remainder <= 0:
                break
            quantities[idx] += 1
            remainder -= 1
    now = datetime.now(timezone.utc)
    slices: list[AlgoSlice] = []
    for idx, qty in enumerate(quantities):
        if qty <= 0:
            continue
        delay = int(duration_seconds * idx / max(len(profile) - 1, 1))
        slices.append(AlgoSlice(qty=qty, earliest_at=now + timedelta(seconds=delay)))
    return slices

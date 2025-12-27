from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


@dataclass
class AlgoSlice:
    qty: int
    earliest_at: datetime


def twap_slices(total_qty: int, duration_seconds: int, slice_count: int) -> list[AlgoSlice]:
    if total_qty <= 0 or slice_count <= 0:
        return []
    now = datetime.utcnow()
    qty_per = max(1, total_qty // slice_count)
    slices: list[AlgoSlice] = []
    for idx in range(slice_count):
        qty = qty_per if idx < slice_count - 1 else max(1, total_qty - qty_per * (slice_count - 1))
        delay = int(duration_seconds * idx / max(slice_count - 1, 1))
        slices.append(AlgoSlice(qty=qty, earliest_at=now + timedelta(seconds=delay)))
    return slices


def pov_slices(total_qty: int, max_participation: float, est_volume: float) -> list[AlgoSlice]:
    if total_qty <= 0:
        return []
    allowed = max(1, int(est_volume * max_participation))
    slices = []
    remaining = total_qty
    now = datetime.utcnow()
    while remaining > 0:
        qty = min(allowed, remaining)
        slices.append(AlgoSlice(qty=qty, earliest_at=now))
        remaining -= qty
    return slices


def vwap_slices(total_qty: int, volume_profile: Iterable[float], duration_seconds: int) -> list[AlgoSlice]:
    profile = list(volume_profile)
    if total_qty <= 0 or not profile:
        return []
    total = sum(profile) or 1.0
    now = datetime.utcnow()
    slices: list[AlgoSlice] = []
    for idx, weight in enumerate(profile):
        qty = max(1, int(total_qty * (weight / total)))
        delay = int(duration_seconds * idx / max(len(profile) - 1, 1))
        slices.append(AlgoSlice(qty=qty, earliest_at=now + timedelta(seconds=delay)))
    # adjust remainder
    diff = total_qty - sum(s.qty for s in slices)
    if diff != 0 and slices:
        slices[-1].qty = max(1, slices[-1].qty + diff)
    return slices

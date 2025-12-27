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
    slice_count = min(slice_count, total_qty)
    now = datetime.utcnow()
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
    now = datetime.utcnow()
    slices: list[AlgoSlice] = []
    for idx, qty in enumerate(quantities):
        if qty <= 0:
            continue
        delay = int(duration_seconds * idx / max(len(profile) - 1, 1))
        slices.append(AlgoSlice(qty=qty, earliest_at=now + timedelta(seconds=delay)))
    return slices

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from datetime import datetime

from app.execution.algos import twap_slices, vwap_slices, pov_slices


def test_twap_slices_sum() -> None:
    slices = twap_slices(total_qty=10, duration_seconds=60, slice_count=4)
    assert sum(s.qty for s in slices) == 10
    assert slices[0].earliest_at <= slices[-1].earliest_at


def test_vwap_slices_sum() -> None:
    slices = vwap_slices(total_qty=9, volume_profile=[1, 2, 1], duration_seconds=60)
    assert sum(s.qty for s in slices) == 9


def test_pov_slices_participation() -> None:
    slices = pov_slices(total_qty=10, max_participation=0.1, est_volume=50)
    assert all(s.qty <= 5 for s in slices)
    assert sum(s.qty for s in slices) == 10

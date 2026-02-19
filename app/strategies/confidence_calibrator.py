# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Bin-based confidence calibrator (no sklearn dependency).

Tracks (raw_confidence, was_profitable) per strategy in a rolling window
and maps raw confidence to empirical win-rate per bin.
"""

from __future__ import annotations

from collections import deque


class ConfidenceCalibrator:
    """Simple histogram-based confidence calibration."""

    def __init__(self, window: int = 200, n_bins: int = 5, min_samples: int = 30):
        self._window = window
        self._n_bins = n_bins
        self._min_samples = min_samples
        # per-strategy rolling history: deque of (raw_confidence, was_profitable)
        self._history: dict[str, deque[tuple[float, bool]]] = {}
        # cached bin edges and win rates
        self._bin_edges: list[float] = [i / n_bins for i in range(n_bins + 1)]

    def record(self, strategy: str, raw_confidence: float, was_profitable: bool) -> None:
        buf = self._history.setdefault(strategy, deque(maxlen=self._window))
        buf.append((max(0.0, min(1.0, raw_confidence)), was_profitable))

    def calibrate(self, strategy: str, raw_confidence: float) -> float:
        raw = max(0.0, min(1.0, raw_confidence))
        buf = self._history.get(strategy)
        if buf is None or len(buf) < self._min_samples:
            return raw * 0.75  # conservative before enough data

        # Compute win rate per bin
        bin_wins: list[int] = [0] * self._n_bins
        bin_counts: list[int] = [0] * self._n_bins
        for conf, won in buf:
            idx = min(int(conf * self._n_bins), self._n_bins - 1)
            bin_counts[idx] += 1
            if won:
                bin_wins[idx] += 1

        # Win rate per bin (with smoothing)
        bin_rates = []
        for w, c in zip(bin_wins, bin_counts):
            if c >= 3:
                bin_rates.append(w / c)
            else:
                bin_rates.append(0.5)  # neutral for sparse bins

        # Linear interpolation within bin
        pos = raw * self._n_bins
        idx = min(int(pos), self._n_bins - 1)
        frac = pos - idx
        if idx < self._n_bins - 1:
            calibrated = bin_rates[idx] * (1 - frac) + bin_rates[idx + 1] * frac
        else:
            calibrated = bin_rates[idx]

        return max(0.0, min(1.0, calibrated))

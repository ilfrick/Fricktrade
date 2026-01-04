# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from app.learning.features import build_observation


def compute_feature_stats(
    datasets: list,
    window_size: int,
    feature_config: dict | None = None,
    max_samples: int = 5000,
    stride: int = 5,
) -> dict[str, Any]:
    if not datasets:
        return {}
    feature_config = feature_config or {}
    sums = None
    sumsq = None
    count = 0
    for df in datasets:
        if count >= max_samples:
            break
        closes = df.get("Close").tolist()
        volumes = df.get("Volume").tolist()
        limit = len(closes)
        for idx in range(window_size, limit):
            if count >= max_samples:
                break
            if stride > 1 and ((idx - window_size) % stride != 0):
                continue
            obs = build_observation(
                closes=closes[: idx + 1],
                volumes=volumes[: idx + 1],
                window_size=window_size,
                position=0.0,
                cash_pct=1.0,
                buying_power_pct=1.0,
                feature_config=feature_config,
            )
            vec = np.asarray(obs, dtype=float).reshape(-1)
            if sums is None:
                sums = np.zeros_like(vec)
                sumsq = np.zeros_like(vec)
            sums += vec
            sumsq += vec**2
            count += 1
    if sums is None or count == 0:
        return {}
    mean = sums / count
    var = (sumsq / count) - (mean**2)
    std = np.sqrt(np.clip(var, 0.0, None))
    return {"mean": mean.tolist(), "std": std.tolist(), "count": count}


class DriftMonitor:
    def __init__(
        self,
        baseline_stats: dict | None,
        window: int = 120,
        feature_zscore_threshold: float = 3.0,
        max_drift_feature_pct: float = 0.3,
        pnl_window: int = 30,
        max_pnl_drop_pct: float = 2.0,
    ):
        baseline_stats = baseline_stats or {}
        self._baseline_mean = np.asarray(baseline_stats.get("mean", []), dtype=float)
        self._baseline_std = np.asarray(baseline_stats.get("std", []), dtype=float)
        self._window = max(int(window), 1)
        self._feature_zscore_threshold = float(feature_zscore_threshold)
        self._max_drift_feature_pct = float(max_drift_feature_pct)
        self._pnl_window = max(int(pnl_window), 1)
        self._max_pnl_drop_pct = float(max_pnl_drop_pct)
        self._feature_history: deque[np.ndarray] = deque(maxlen=self._window)
        self._pnl_history: deque[float] = deque(maxlen=self._pnl_window)
        self._drifted = False
        self._reasons: list[str] = []

    def update_features(self, obs: np.ndarray) -> None:
        if self._baseline_mean.size == 0:
            return
        vec = np.asarray(obs, dtype=float).reshape(-1)
        if vec.size != self._baseline_mean.size:
            return
        self._feature_history.append(vec)

    def update_pnl(self, pnl_pct: float) -> None:
        self._pnl_history.append(float(pnl_pct))

    def check_drift(self) -> list[str]:
        if self._drifted:
            return []
        reasons = []
        if self._feature_drifted():
            reasons.append("feature_drift")
        if self._pnl_drifted():
            reasons.append("pnl_drift")
        if reasons:
            self._drifted = True
            self._reasons = reasons
        return reasons

    @property
    def drifted(self) -> bool:
        return self._drifted

    @property
    def reasons(self) -> list[str]:
        return list(self._reasons)

    def _feature_drifted(self) -> bool:
        if self._baseline_mean.size == 0:
            return False
        if len(self._feature_history) < self._window:
            return False
        matrix = np.stack(self._feature_history, axis=0)
        mean = matrix.mean(axis=0)
        std = np.where(self._baseline_std == 0.0, 1.0, self._baseline_std)
        zscores = np.abs((mean - self._baseline_mean) / std)
        drift_pct = float(np.mean(zscores > self._feature_zscore_threshold))
        return drift_pct >= self._max_drift_feature_pct

    def _pnl_drifted(self) -> bool:
        if len(self._pnl_history) < self._pnl_window:
            return False
        avg_pnl = sum(self._pnl_history) / len(self._pnl_history)
        return avg_pnl <= -self._max_pnl_drop_pct

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_state: dict) -> dict:
        raise NotImplementedError

    @staticmethod
    def _wilder_rsi(prices: list, period: int = 14) -> float:
        """Wilder's smoothed RSI (correct implementation)."""
        import numpy as np
        if len(prices) < period + 1:
            return 50.0
        closes = np.array(prices, dtype=float)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Seed with simple mean for first period
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        # Wilder smoothing for remaining bars
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

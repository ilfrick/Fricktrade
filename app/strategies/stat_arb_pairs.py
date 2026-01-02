# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from app.strategies.base import Strategy


@dataclass
class PairsParams:
    lookback: int = 50
    z_entry: float = 1.5
    z_exit: float = 0.5
    refresh_minutes: int = 30
    max_pairs: int = 25


class StatArbPairsStrategy(Strategy):
    _price_cache: dict[str, list[float]] = {}
    _pairs: list[tuple[str, str]] = []
    _last_refresh: datetime | None = None

    def __init__(self, params: dict):
        cfg = params.get("stat_arb_pairs", {}) if isinstance(params, dict) else {}
        self.params = PairsParams(
            lookback=int(cfg.get("lookback", 50)),
            z_entry=float(cfg.get("z_entry", 1.5)),
            z_exit=float(cfg.get("z_exit", 0.5)),
            refresh_minutes=int(cfg.get("refresh_minutes", 30)),
            max_pairs=int(cfg.get("max_pairs", 25)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        symbol = market_state.get("symbol")
        prices = market_state.get("prices", []) or []
        if not symbol or len(prices) < 3:
            return {"action": "hold"}
        self._update_cache(symbol, prices)
        self._refresh_pairs_if_needed()
        pair = self._find_pair(symbol)
        if pair is None:
            return {"action": "hold"}
        other = pair[1] if pair[0] == symbol else pair[0]
        series_a = self._price_cache.get(symbol, [])
        series_b = self._price_cache.get(other, [])
        if len(series_a) < self.params.lookback or len(series_b) < self.params.lookback:
            return {"action": "hold"}
        a = np.array(series_a[-self.params.lookback :], dtype=float)
        b = np.array(series_b[-self.params.lookback :], dtype=float)
        if a.std() == 0 or b.std() == 0:
            return {"action": "hold"}
        spread = a - b
        z = (spread[-1] - spread.mean()) / (spread.std() or 1.0)
        if z >= self.params.z_entry:
            return {"action": "sell", "pair": other, "z_score": float(z)}
        if z <= -self.params.z_entry:
            return {"action": "buy", "pair": other, "z_score": float(z)}
        if abs(z) <= self.params.z_exit:
            return {"action": "exit", "pair": other, "z_score": float(z)}
        return {"action": "hold", "pair": other, "z_score": float(z)}

    def _update_cache(self, symbol: str, prices: list[float]) -> None:
        cache = self._price_cache.setdefault(symbol, [])
        if prices:
            cache.append(float(prices[-1]))
        if len(cache) > self.params.lookback * 3:
            del cache[: len(cache) - self.params.lookback * 2]

    def _refresh_pairs_if_needed(self) -> None:
        now = datetime.utcnow()
        if self._last_refresh and now - self._last_refresh < timedelta(minutes=self.params.refresh_minutes):
            return
        symbols = list(self._price_cache.keys())
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                a = self._price_cache.get(symbols[i], [])
                b = self._price_cache.get(symbols[j], [])
                if len(a) < self.params.lookback or len(b) < self.params.lookback:
                    continue
                a_series = np.array(a[-self.params.lookback :], dtype=float)
                b_series = np.array(b[-self.params.lookback :], dtype=float)
                if a_series.std() == 0 or b_series.std() == 0:
                    continue
                corr = float(np.corrcoef(a_series, b_series)[0, 1])
                pairs.append((symbols[i], symbols[j], corr))
        pairs.sort(key=lambda p: abs(p[2]), reverse=True)
        self._pairs = [(a, b) for a, b, _ in pairs[: self.params.max_pairs]]
        self._last_refresh = now

    def _find_pair(self, symbol: str) -> tuple[str, str] | None:
        for pair in self._pairs:
            if symbol in pair:
                return pair
        return None

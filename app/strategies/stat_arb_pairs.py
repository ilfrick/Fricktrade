# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from app.strategies.base import Strategy


@dataclass
class PairsParams:
    lookback: int = 50
    z_entry: float = 2.0
    z_exit: float = 0.5
    refresh_minutes: int = 30
    max_pairs: int = 25


class StatArbPairsStrategy(Strategy):
    # Class-level shared state — all per-symbol instances share the same price cache
    # and pair list so pair discovery works across symbols (required for cointegration)
    _shared_price_cache: dict[str, list[float]] = {}
    _shared_pairs: list[tuple[str, str, float]] = []
    _shared_last_refresh: datetime | None = None
    _shared_lock: threading.Lock = threading.Lock()

    def __init__(self, params: dict):
        cfg = params.get("stat_arb_pairs", {}) if isinstance(params, dict) else {}
        self.params = PairsParams(
            lookback=int(cfg.get("lookback", 50)),
            z_entry=float(cfg.get("z_entry", 2.0)),
            z_exit=float(cfg.get("z_exit", 0.5)),
            refresh_minutes=int(cfg.get("refresh_minutes", 30)),
            max_pairs=int(cfg.get("max_pairs", 25)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        symbol = market_state.get("symbol", "")
        if "/" in (symbol or ""):
            return {"action": "hold", "confidence": 0.0, "reason": "equity_only", "name": "stat_arb_pairs"}
        prices = market_state.get("prices", []) or []
        if not symbol or len(prices) < 3:
            return {"action": "hold", "name": "stat_arb_pairs"}
        self._update_cache(symbol, prices)
        self._refresh_pairs_if_needed()
        pair_info = self._find_pair(symbol)

        if pair_info is None:
            return {"action": "hold", "name": "stat_arb_pairs"}
        sym_a, sym_b, hedge_ratio = pair_info
        other = sym_b if sym_a == symbol else sym_a
        with StatArbPairsStrategy._shared_lock:
            series_a = list(StatArbPairsStrategy._shared_price_cache.get(sym_a, []))
            series_b = list(StatArbPairsStrategy._shared_price_cache.get(sym_b, []))
        if len(series_a) < self.params.lookback or len(series_b) < self.params.lookback:
            return {"action": "hold", "name": "stat_arb_pairs"}
        a = np.array(series_a[-self.params.lookback :], dtype=float)
        b = np.array(series_b[-self.params.lookback :], dtype=float)
        if np.any(a <= 0) or np.any(b <= 0):
            return {"action": "hold", "name": "stat_arb_pairs"}
        # Log-ratio spread: log(a) - beta * log(b)
        log_a = np.log(a)
        log_b = np.log(b)
        spread = log_a - hedge_ratio * log_b
        spread_mean = float(spread.mean())
        spread_std = float(spread.std())
        if spread_std < 1e-10:
            return {"action": "hold", "name": "stat_arb_pairs"}
        z = (spread[-1] - spread_mean) / spread_std

        # Flip signal if current symbol is sym_b (the hedge leg)
        if symbol == sym_b:
            z = -z

        confidence = float(np.clip(abs(z) / (self.params.z_entry * 1.5), 0.0, 1.0))
        if z >= self.params.z_entry:
            return {"action": "sell", "name": "stat_arb_pairs", "pair": other, "z_score": float(z), "confidence": confidence}
        if z <= -self.params.z_entry:
            return {"action": "buy", "name": "stat_arb_pairs", "pair": other, "z_score": float(z), "confidence": confidence}
        if abs(z) <= self.params.z_exit:
            return {"action": "exit", "name": "stat_arb_pairs", "pair": other, "z_score": float(z), "confidence": confidence}
        return {"action": "hold", "name": "stat_arb_pairs", "pair": other, "z_score": float(z)}

    def _update_cache(self, symbol: str, prices: list[float]) -> None:
        with StatArbPairsStrategy._shared_lock:
            cache = StatArbPairsStrategy._shared_price_cache.setdefault(symbol, [])
            if prices:
                cache.append(float(prices[-1]))
            if len(cache) > self.params.lookback * 3:
                del cache[: len(cache) - self.params.lookback * 2]

    def _refresh_pairs_if_needed(self) -> None:
        now = datetime.now(timezone.utc)
        with StatArbPairsStrategy._shared_lock:
            last = StatArbPairsStrategy._shared_last_refresh
            if last and now - last < timedelta(minutes=self.params.refresh_minutes):
                return
            # Claim the refresh slot before releasing the lock so concurrent callers
            # see a recent timestamp and skip their own redundant refresh (TOCTOU fix).
            StatArbPairsStrategy._shared_last_refresh = now
            symbols = list(StatArbPairsStrategy._shared_price_cache.keys())
            # Snapshot series under lock, do heavy computation outside
            snapshots = {
                s: list(StatArbPairsStrategy._shared_price_cache[s])
                for s in symbols
            }
        candidates: list[tuple[str, str, float, float]] = []  # (sym_a, sym_b, hedge_ratio, t_stat)
        n_tested = 0
        n_passed = 0
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                a = snapshots.get(symbols[i], [])
                b = snapshots.get(symbols[j], [])
                if len(a) < self.params.lookback or len(b) < self.params.lookback:
                    continue
                a_series = np.array(a[-self.params.lookback :], dtype=float)
                b_series = np.array(b[-self.params.lookback :], dtype=float)
                if np.any(a_series <= 0) or np.any(b_series <= 0):
                    continue
                n_tested += 1
                hedge_ratio, t_stat = self._cointegration_test(a_series, b_series)
                if t_stat is not None and t_stat < -2.86:  # ~5% significance for ADF
                    n_passed += 1
                    candidates.append((symbols[i], symbols[j], hedge_ratio, t_stat))
        # Rank by most negative t-stat (strongest cointegration)
        candidates.sort(key=lambda p: p[3])
        new_pairs = [(a, b, hr) for a, b, hr, _ in candidates[: self.params.max_pairs]]
        with StatArbPairsStrategy._shared_lock:
            StatArbPairsStrategy._shared_pairs = new_pairs
            # _shared_last_refresh already set optimistically above; no need to re-set here
        logging.info(
            "stat_arb: %d symbols in cache, %d pairs tested, %d passed ADF, %d active pairs",
            len(symbols), n_tested, n_passed, len(new_pairs),
        )

    def _find_pair(self, symbol: str) -> tuple[str, str, float] | None:
        with StatArbPairsStrategy._shared_lock:
            pairs = StatArbPairsStrategy._shared_pairs
        for pair in pairs:
            if symbol in (pair[0], pair[1]):
                return pair
        return None

    @staticmethod
    def _cointegration_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float | None]:
        """Simplified ADF test using numpy only.

        Returns (hedge_ratio, t_stat). t_stat < -2.86 suggests cointegration at 5%.
        """
        log_a = np.log(a)
        log_b = np.log(b)
        # OLS hedge ratio: log_a = beta * log_b + alpha
        beta = float(np.polyfit(log_b, log_a, 1)[0])
        # Spread residuals
        spread = log_a - beta * log_b
        # ADF test on spread: regress delta_spread on lagged_spread
        n = len(spread)
        if n < 10:
            return beta, None
        delta = np.diff(spread)
        lagged = spread[:-1]
        # Simple OLS: delta = gamma * lagged + error
        # gamma = cov(delta, lagged) / var(lagged)
        lagged_mean = float(lagged.mean())
        delta_mean = float(delta.mean())
        lagged_centered = lagged - lagged_mean
        var_lagged = float((lagged_centered ** 2).sum())
        if var_lagged < 1e-15:
            return beta, None
        gamma = float(((delta - delta_mean) * lagged_centered).sum() / var_lagged)
        # Residuals for standard error
        residuals = delta - (gamma * lagged + (delta_mean - gamma * lagged_mean))
        sse = float((residuals ** 2).sum())
        se_gamma = (sse / max(n - 3, 1) / max(var_lagged, 1e-15)) ** 0.5
        if se_gamma < 1e-15:
            return beta, None
        t_stat = gamma / se_gamma
        return beta, t_stat

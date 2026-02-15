# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.strategies.base import Strategy


@dataclass
class FactorParams:
    momentum_weight: float = 0.5
    liquidity_weight: float = 0.2
    volatility_weight: float = 0.1
    mr_weight: float = 0.15
    buy_threshold: float = 0.2
    sell_threshold: float = -0.2


class FactorModelStrategy(Strategy):
    def __init__(self, params: dict):
        cfg = params.get("factor_model", {}) if isinstance(params, dict) else {}
        self.params = FactorParams(
            momentum_weight=float(cfg.get("momentum_weight", 0.5)),
            liquidity_weight=float(cfg.get("liquidity_weight", 0.2)),
            volatility_weight=float(cfg.get("volatility_weight", 0.1)),
            mr_weight=float(cfg.get("mr_weight", 0.15)),
            buy_threshold=float(cfg.get("buy_threshold", 0.2)),
            sell_threshold=float(cfg.get("sell_threshold", -0.2)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        prices = market_state.get("prices", []) or []
        volumes = market_state.get("volumes", []) or []
        if len(prices) < 3:
            return {"action": "hold"}
        if any(price <= 0 for price in prices):
            return {"action": "hold"}
        close = np.array(prices, dtype=float)
        returns = np.diff(close) / close[:-1]

        # Momentum: extended to 10 bars (was 3)
        lookback = min(10, returns.size)
        momentum = float(returns[-lookback:].mean()) if returns.size else 0.0

        # Liquidity score
        liquidity = float(sum(volumes[-3:])) if volumes else 0.0
        liquidity_score = np.tanh(liquidity / 1_000_000.0)

        # Volatility score
        vol = float(returns[-5:].std()) if returns.size >= 5 else float(returns.std()) if returns.size else 0.0
        vol_score = 1.0 - np.tanh(vol * 10.0)

        # Mean-reversion factor: z-score of price vs 20-bar mean (inverted)
        mr_window = min(20, len(close))
        if mr_window >= 5:
            price_mean = float(close[-mr_window:].mean())
            price_std = float(close[-mr_window:].std())
            mr_score = float((close[-1] - price_mean) / price_std) if price_std > 0 else 0.0
            mr_score = float(np.clip(-mr_score * 0.1, -1.0, 1.0))  # inverted: oversold=positive
        else:
            mr_score = 0.0

        # Trend quality gate: simplified ADX proxy
        if returns.size >= 14 and len(close) >= 15:
            abs_returns = np.abs(returns[-14:])
            net_move = abs(float(close[-1] - close[-15]))
            total_move = float(abs_returns.sum())
            trend_quality = net_move / total_move if total_move > 0 else 0.0
        else:
            trend_quality = 0.5  # neutral

        # Composite score with mean-reversion factor
        score = (
            self.params.momentum_weight * momentum
            + self.params.liquidity_weight * liquidity_score
            + self.params.volatility_weight * vol_score
            + self.params.mr_weight * mr_score
        )

        # Trend quality gate: avoid choppy markets
        if trend_quality < 0.3:
            return {"action": "hold", "score": float(score), "confidence": 0.0, "trend_quality": trend_quality}

        max_score = max(abs(self.params.buy_threshold), abs(self.params.sell_threshold)) * 3.0
        confidence = float(min(abs(score) / max(max_score, 0.01), 1.0))
        if score >= self.params.buy_threshold:
            return {"action": "buy", "score": float(score), "confidence": confidence, "trend_quality": trend_quality}
        if score <= self.params.sell_threshold:
            return {"action": "sell", "score": float(score), "confidence": confidence, "trend_quality": trend_quality}
        return {"action": "hold", "score": float(score), "confidence": 0.0, "trend_quality": trend_quality}

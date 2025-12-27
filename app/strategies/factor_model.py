from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.strategies.base import Strategy


@dataclass
class FactorParams:
    momentum_weight: float = 0.6
    liquidity_weight: float = 0.3
    volatility_weight: float = 0.1
    buy_threshold: float = 0.2
    sell_threshold: float = -0.2


class FactorModelStrategy(Strategy):
    def __init__(self, params: dict):
        cfg = params.get("factor_model", {}) if isinstance(params, dict) else {}
        self.params = FactorParams(
            momentum_weight=float(cfg.get("momentum_weight", 0.6)),
            liquidity_weight=float(cfg.get("liquidity_weight", 0.3)),
            volatility_weight=float(cfg.get("volatility_weight", 0.1)),
            buy_threshold=float(cfg.get("buy_threshold", 0.2)),
            sell_threshold=float(cfg.get("sell_threshold", -0.2)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        prices = market_state.get("prices", []) or []
        volumes = market_state.get("volumes", []) or []
        if len(prices) < 3:
            return {"action": "hold"}
        close = np.array(prices, dtype=float)
        returns = np.diff(close) / close[:-1]
        momentum = float(returns[-3:].mean()) if returns.size else 0.0
        liquidity = float(sum(volumes[-3:])) if volumes else 0.0
        liquidity_score = np.tanh(liquidity / 1_000_000.0)
        vol = float(returns[-5:].std()) if returns.size >= 5 else float(returns.std()) if returns.size else 0.0
        vol_score = 1.0 - np.tanh(vol * 10.0)
        score = (
            self.params.momentum_weight * momentum
            + self.params.liquidity_weight * liquidity_score
            + self.params.volatility_weight * vol_score
        )
        if score >= self.params.buy_threshold:
            return {"action": "buy", "score": float(score)}
        if score <= self.params.sell_threshold:
            return {"action": "sell", "score": float(score)}
        return {"action": "hold", "score": float(score)}

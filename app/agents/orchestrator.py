from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev


@dataclass
class OrchestratorConfig:
    enabled: bool = False
    mode: str = "select"
    top_k: int = 2
    min_score: float = 0.0
    normalize_scores: bool = True
    strategy_weights: dict[str, dict[str, float]] | None = None


class StrategyOrchestrator:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.cfg = OrchestratorConfig(
            enabled=bool(cfg.get("enabled", False)),
            mode=str(cfg.get("mode", "select")),
            top_k=int(cfg.get("top_k", 2)),
            min_score=float(cfg.get("min_score", 0.0)),
            normalize_scores=bool(cfg.get("normalize_scores", True)),
            strategy_weights=cfg.get("strategy_weights") or {},
        )

    def select(
        self, strategy_names: list[str], market_state: dict
    ) -> tuple[list[str], dict[str, float]]:
        if not self.cfg.enabled or not strategy_names:
            return strategy_names, {name: 1.0 for name in strategy_names}

        features = _extract_features(market_state)
        scores: dict[str, float] = {}
        for name in strategy_names:
            weights = self.cfg.strategy_weights.get(name, {})
            score = 0.0
            weight_sum = 0.0
            for feature_name, value in features.items():
                weight = float(weights.get(feature_name, 0.0))
                score += weight * value
                weight_sum += abs(weight)
            if self.cfg.normalize_scores and weight_sum > 0:
                score /= weight_sum
            scores[name] = score

        sorted_names = sorted(strategy_names, key=lambda n: scores.get(n, 0.0), reverse=True)
        if self.cfg.mode == "weight":
            weights = {name: max(scores.get(name, 0.0), 0.0) for name in sorted_names}
            if not any(weight > 0 for weight in weights.values()):
                weights = {name: 1.0 for name in sorted_names}
            return sorted_names, weights

        selected = []
        for name in sorted_names:
            if scores.get(name, 0.0) < self.cfg.min_score:
                continue
            selected.append(name)
            if self.cfg.top_k > 0 and len(selected) >= self.cfg.top_k:
                break

        if not selected:
            return strategy_names, {name: 1.0 for name in strategy_names}
        return selected, {name: 1.0 for name in selected}


def _extract_features(market_state: dict) -> dict[str, float]:
    prices = market_state.get("prices") or []
    returns = []
    for idx in range(1, len(prices)):
        prev = prices[idx - 1]
        curr = prices[idx]
        if prev:
            returns.append((curr - prev) / prev * 100.0)

    momentum = returns[-1] if returns else 0.0
    trend = 0.0
    if prices:
        first = prices[0]
        last = prices[-1]
        if first:
            trend = (last - first) / first * 100.0
    volatility = pstdev(returns) if len(returns) > 1 else 0.0

    return {
        "momentum": momentum,
        "trend": trend,
        "volatility": volatility,
        "relative_volume": float(market_state.get("relative_volume", 0.0) or 0.0),
        "session_gain_pct": float(market_state.get("session_gain_pct", 0.0) or 0.0),
        "spread": float(market_state.get("spread_pct", 0.0) or 0.0),
        "catalyst": 1.0 if market_state.get("catalyst") else 0.0,
    }

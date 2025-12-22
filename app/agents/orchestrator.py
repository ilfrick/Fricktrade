from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from statistics import pstdev


@dataclass
class OrchestratorConfig:
    enabled: bool = False
    mode: str = "select"
    top_k: int = 2
    min_score: float = 0.0
    normalize_scores: bool = True
    strategy_weights: dict[str, dict[str, float]] | None = None
    learning_enabled: bool = False
    learning_rate: float = 0.1
    min_bias: float = -1.0
    max_bias: float = 1.0
    decay: float = 0.0
    min_price_move_pct: float = 0.05
    state_path: str = "/data/orchestrator_state.json"
    save_interval_seconds: int = 300
    use_best_state: bool = True
    best_state_path: str = "/data/orchestrator_state_best.json"
    best_score_path: str = "/data/orchestrator_best_score.json"
    score_ema_alpha: float = 0.1


class StrategyOrchestrator:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        learning_cfg = cfg.get("learning", {})
        self.cfg = OrchestratorConfig(
            enabled=bool(cfg.get("enabled", False)),
            mode=str(cfg.get("mode", "select")),
            top_k=int(cfg.get("top_k", 2)),
            min_score=float(cfg.get("min_score", 0.0)),
            normalize_scores=bool(cfg.get("normalize_scores", True)),
            strategy_weights=cfg.get("strategy_weights") or {},
            learning_enabled=bool(learning_cfg.get("enabled", False)),
            learning_rate=float(learning_cfg.get("learning_rate", 0.1)),
            min_bias=float(learning_cfg.get("min_bias", -1.0)),
            max_bias=float(learning_cfg.get("max_bias", 1.0)),
            decay=float(learning_cfg.get("decay", 0.0)),
            min_price_move_pct=float(learning_cfg.get("min_price_move_pct", 0.05)),
            state_path=str(learning_cfg.get("state_path", "/data/orchestrator_state.json")),
            save_interval_seconds=int(learning_cfg.get("save_interval_seconds", 300)),
            use_best_state=bool(learning_cfg.get("use_best_state", True)),
            best_state_path=str(learning_cfg.get("best_state_path", "/data/orchestrator_state_best.json")),
            best_score_path=str(learning_cfg.get("best_score_path", "/data/orchestrator_best_score.json")),
            score_ema_alpha=float(learning_cfg.get("score_ema_alpha", 0.1)),
        )
        self._biases: dict[str, float] = {}
        self._last_saved_at: float | None = None
        self._score_ema: float | None = None
        self._best_score: float | None = None
        self._load_state()

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
            scores[name] = score + self._biases.get(name, 0.0)

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

    def update_biases(self, decisions: dict[str, str], prev_price: float, current_price: float) -> None:
        if not self.cfg.learning_enabled:
            return
        if prev_price <= 0 or current_price <= 0:
            return
        move_pct = (current_price - prev_price) / prev_price * 100.0
        if abs(move_pct) < self.cfg.min_price_move_pct:
            return
        direction = 1.0 if move_pct > 0 else -1.0
        decay = max(0.0, min(self.cfg.decay, 1.0))
        lr = self.cfg.learning_rate
        for name, action in decisions.items():
            reward = 0.0
            if action == "buy":
                reward = direction
            elif action == "sell":
                reward = -direction
            else:
                continue
            bias = self._biases.get(name, 0.0)
            bias = bias * (1.0 - decay) + lr * reward
            bias = max(self.cfg.min_bias, min(self.cfg.max_bias, bias))
            self._biases[name] = bias
        self._update_score(decisions, direction)
        self._maybe_save_state()
        self._maybe_save_best()

    def _load_state(self) -> None:
        best_path = Path(self.cfg.best_state_path)
        if self.cfg.use_best_state and best_path.exists():
            if self._load_biases(best_path):
                self._load_best_score()
                return
        path = Path(self.cfg.state_path)
        self._load_biases(path)
        self._load_best_score()

    def _load_biases(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        biases = raw.get("biases", {})
        if isinstance(biases, dict):
            self._biases = {str(k): float(v) for k, v in biases.items()}
            return True
        return False

    def _load_best_score(self) -> None:
        path = Path(self.cfg.best_score_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        score = raw.get("best_score")
        if isinstance(score, (int, float)):
            self._best_score = float(score)

    def _update_score(self, decisions: dict[str, str], direction: float) -> None:
        rewards = []
        for action in decisions.values():
            if action == "buy":
                rewards.append(direction)
            elif action == "sell":
                rewards.append(-direction)
        if not rewards:
            return
        avg_reward = sum(rewards) / len(rewards)
        alpha = self.cfg.score_ema_alpha
        if self._score_ema is None:
            self._score_ema = avg_reward
        else:
            self._score_ema = alpha * avg_reward + (1.0 - alpha) * self._score_ema

    def _maybe_save_state(self) -> None:
        now = time.time()
        if self._last_saved_at and now - self._last_saved_at < self.cfg.save_interval_seconds:
            return
        path = Path(self.cfg.state_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"biases": self._biases, "saved_at": now}
            path.write_text(json.dumps(payload, indent=2))
            self._last_saved_at = now
        except OSError:
            return

    def _maybe_save_best(self) -> None:
        if self._score_ema is None:
            return
        if self._best_score is not None and self._score_ema <= self._best_score:
            return
        self._best_score = self._score_ema
        best_state_path = Path(self.cfg.best_state_path)
        best_score_path = Path(self.cfg.best_score_path)
        try:
            best_state_path.parent.mkdir(parents=True, exist_ok=True)
            best_state_path.write_text(json.dumps({"biases": self._biases, "saved_at": time.time()}, indent=2))
            best_score_path.write_text(json.dumps({"best_score": self._best_score, "saved_at": time.time()}, indent=2))
        except OSError:
            return


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

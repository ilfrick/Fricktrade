from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import random
import time
from pathlib import Path
from statistics import pstdev

import torch
from torch import nn

import yfinance as yf


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


@dataclass
class MLOrchestratorConfig:
    enabled: bool = False
    device: str = "auto"
    hidden_dim: int = 64
    dropout: float = 0.1
    model_path: str = "/data/orchestrator_model.pt"
    best_model_path: str = "/data/orchestrator_model_best.pt"
    use_best_model: bool = True
    best_score_path: str = "/data/orchestrator_model_best_score.json"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 5000
    update_steps_per_bar: int = 1
    epsilon: float = 0.05
    min_price_move_pct: float = 0.02
    reward_scale: float = 1.0
    max_grad_norm: float = 1.0
    save_interval_seconds: int = 300
    score_ema_alpha: float = 0.1
    pretrain_enabled: bool = True
    pretrain_lookback_days: int = 30
    pretrain_interval: str = "5m"
    pretrain_max_symbols: int = 20
    pretrain_max_samples: int = 2000
    pretrain_epochs: int = 2
    pretrain_warmup_bars: int = 50
    pretrain_symbols_source: str = "data"

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


class MLStrategyOrchestrator:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        ml_cfg = cfg.get("ml", {})
        self._mode = str(cfg.get("mode", "select"))
        self._top_k = int(cfg.get("top_k", 2))
        self._min_score = float(cfg.get("min_score", 0.0))
        self.cfg = MLOrchestratorConfig(
            enabled=bool(ml_cfg.get("enabled", False)),
            device=str(ml_cfg.get("device", "auto")),
            hidden_dim=int(ml_cfg.get("hidden_dim", 64)),
            dropout=float(ml_cfg.get("dropout", 0.1)),
            model_path=str(ml_cfg.get("model_path", "/data/orchestrator_model.pt")),
            best_model_path=str(ml_cfg.get("best_model_path", "/data/orchestrator_model_best.pt")),
            use_best_model=bool(ml_cfg.get("use_best_model", True)),
            best_score_path=str(ml_cfg.get("best_score_path", "/data/orchestrator_model_best_score.json")),
            learning_rate=float(ml_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(ml_cfg.get("weight_decay", 1e-4)),
            batch_size=int(ml_cfg.get("batch_size", 64)),
            buffer_size=int(ml_cfg.get("buffer_size", 5000)),
            update_steps_per_bar=int(ml_cfg.get("update_steps_per_bar", 1)),
            epsilon=float(ml_cfg.get("epsilon", 0.05)),
            min_price_move_pct=float(ml_cfg.get("min_price_move_pct", 0.02)),
            reward_scale=float(ml_cfg.get("reward_scale", 1.0)),
            max_grad_norm=float(ml_cfg.get("max_grad_norm", 1.0)),
            save_interval_seconds=int(ml_cfg.get("save_interval_seconds", 300)),
            score_ema_alpha=float(ml_cfg.get("score_ema_alpha", 0.1)),
            pretrain_enabled=bool(ml_cfg.get("pretrain", {}).get("enabled", True)),
            pretrain_lookback_days=int(ml_cfg.get("pretrain", {}).get("lookback_days", 30)),
            pretrain_interval=str(ml_cfg.get("pretrain", {}).get("interval", "5m")),
            pretrain_max_symbols=int(ml_cfg.get("pretrain", {}).get("max_symbols", 20)),
            pretrain_max_samples=int(ml_cfg.get("pretrain", {}).get("max_samples", 2000)),
            pretrain_epochs=int(ml_cfg.get("pretrain", {}).get("epochs", 2)),
            pretrain_warmup_bars=int(ml_cfg.get("pretrain", {}).get("warmup_bars", 50)),
            pretrain_symbols_source=str(ml_cfg.get("pretrain", {}).get("symbols_source", "data")),
        )
        self._device = _resolve_device(self.cfg.device)
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self._strategy_names: list[str] = []
        self._buffer: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=self.cfg.buffer_size)
        self._last_state: dict[str, dict[str, object]] = {}
        self._last_saved_at: float | None = None
        self._score_ema: float | None = None
        self._best_score: float | None = None

    def is_enabled(self) -> bool:
        return self.cfg.enabled

    def bootstrap(self, strategy_names: list[str], build_strategy, strategy_params: dict, data_cfg: dict) -> None:
        if not self.cfg.enabled:
            return
        self._ensure_model(strategy_names)
        self._load_best_score()
        if self._load_model():
            return
        if not self.cfg.pretrain_enabled:
            return
        self._pretrain(strategy_names, build_strategy, strategy_params, data_cfg)

    def select(
        self,
        strategy_names: list[str],
        market_state: dict,
        signals: list[dict] | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        if not self.cfg.enabled or not strategy_names:
            return strategy_names, {name: 1.0 for name in strategy_names}
        self._ensure_model(strategy_names)
        features = _feature_vector(market_state)
        scores = self._predict(features)
        if random.random() < self.cfg.epsilon:
            shuffled = strategy_names[:]
            random.shuffle(shuffled)
            selected = shuffled[: max(1, min(len(shuffled), self._top_k))]
            return selected, {name: 1.0 for name in selected}

        ranked = sorted(strategy_names, key=lambda n: scores.get(n, 0.0), reverse=True)
        if not ranked:
            return strategy_names, {name: 1.0 for name in strategy_names}
        if len(ranked) == 1:
            return ranked, {ranked[0]: 1.0}

        selected = [name for name in ranked if scores.get(name, 0.0) >= self._min_score]
        if not selected:
            selected = ranked
        top_k = max(1, min(len(selected), self._top_k))
        selected = selected[:top_k]
        if self._mode == "weight":
            weights = {name: max(scores.get(name, 0.0), 0.0) for name in selected}
            if not any(weight > 0 for weight in weights.values()):
                weights = {name: 1.0 for name in selected}
            return selected, weights
        return selected, {name: 1.0 for name in selected}

    def record(self, symbol: str, signals: list[dict], market_state: dict) -> None:
        if not self.cfg.enabled:
            return
        actions = {s.get("name"): s.get("action") for s in signals if s.get("name")}
        if not actions:
            return
        last_price = _last_price(market_state)
        if last_price is None:
            return
        features = _feature_vector(market_state)
        self._last_state[symbol] = {"features": features, "actions": actions, "price": last_price}

    def update(self, symbol: str, market_state: dict) -> None:
        if not self.cfg.enabled:
            return
        state = self._last_state.get(symbol)
        if not state:
            return
        prev_price = state.get("price")
        if prev_price is None:
            return
        current_price = _last_price(market_state)
        if current_price is None:
            return
        move_pct = (current_price - prev_price) / prev_price * 100.0
        if abs(move_pct) < self.cfg.min_price_move_pct:
            self._last_state.pop(symbol, None)
            return
        rewards = {}
        for name, action in state.get("actions", {}).items():
            if action == "buy":
                reward = move_pct
            elif action == "sell":
                reward = -move_pct
            else:
                reward = 0.0
            rewards[name] = reward * self.cfg.reward_scale
        self._enqueue(state.get("features"), rewards)
        self._train()
        self._update_score(rewards)
        self._maybe_save()
        self._maybe_save_best()
        self._last_state.pop(symbol, None)

    def _ensure_model(self, strategy_names: list[str]) -> None:
        if self._model is not None:
            return
        self._strategy_names = strategy_names
        self._input_dim = len(_feature_vector({}))
        self._output_dim = len(strategy_names)
        self._model = _MLP(self._input_dim, self._output_dim, self.cfg.hidden_dim, self.cfg.dropout).to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def _predict(self, features: list[float]) -> dict[str, float]:
        if not self._model:
            return {name: 0.0 for name in self._strategy_names}
        vec = torch.tensor(features, dtype=torch.float32, device=self._device).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            scores = self._model(vec).squeeze(0).cpu().tolist()
        return {name: float(score) for name, score in zip(self._strategy_names, scores)}

    def _enqueue(self, features: list[float] | None, rewards: dict[str, float]) -> None:
        if features is None or not rewards:
            return
        target = [rewards.get(name, 0.0) for name in self._strategy_names]
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        self._buffer.append((x, y))

    def _train(self) -> None:
        if not self._model or not self._optimizer:
            return
        if len(self._buffer) < self.cfg.batch_size:
            return
        self._model.train()
        for _ in range(max(1, self.cfg.update_steps_per_bar)):
            batch = random.sample(list(self._buffer), self.cfg.batch_size)
            x = torch.stack([item[0] for item in batch]).to(self._device)
            y = torch.stack([item[1] for item in batch]).to(self._device)
            pred = self._model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.cfg.max_grad_norm)
            self._optimizer.step()

    def _update_score(self, rewards: dict[str, float]) -> None:
        values = [v for v in rewards.values() if v is not None]
        if not values:
            return
        avg_reward = sum(values) / len(values)
        alpha = self.cfg.score_ema_alpha
        if self._score_ema is None:
            self._score_ema = avg_reward
        else:
            self._score_ema = alpha * avg_reward + (1.0 - alpha) * self._score_ema

    def _maybe_save(self) -> None:
        now = time.time()
        if self._last_saved_at and now - self._last_saved_at < self.cfg.save_interval_seconds:
            return
        self._save_model(self.cfg.model_path)
        self._last_saved_at = now

    def _maybe_save_best(self) -> None:
        if self._score_ema is None:
            return
        if self._best_score is not None and self._score_ema <= self._best_score:
            return
        self._best_score = self._score_ema
        self._save_model(self.cfg.best_model_path)
        try:
            Path(self.cfg.best_score_path).write_text(
                json.dumps({"best_score": self._best_score, "saved_at": time.time()}, indent=2)
            )
        except OSError:
            return

    def _save_model(self, path: str) -> None:
        if not self._model:
            return
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), target)
        except OSError:
            return

    def _load_model(self) -> bool:
        model_path = self.cfg.best_model_path if self.cfg.use_best_model else self.cfg.model_path
        path = Path(model_path)
        if not path.exists():
            return False
        if not self._model:
            return False
        try:
            self._model.load_state_dict(torch.load(path, map_location=self._device))
            return True
        except OSError:
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

    def _pretrain(self, strategy_names: list[str], build_strategy, strategy_params: dict, data_cfg: dict) -> None:
        symbols = data_cfg.get("symbols", [])
        if self.cfg.pretrain_symbols_source == "data" and symbols:
            symbols = symbols[: self.cfg.pretrain_max_symbols]
        else:
            symbols = symbols[: self.cfg.pretrain_max_symbols]
        if not symbols:
            return
        samples = 0
        strategies = {name: build_strategy(name, strategy_params) for name in strategy_names}
        for symbol in symbols:
            try:
                data = yf.download(
                    tickers=symbol,
                    period=f"{self.cfg.pretrain_lookback_days}d",
                    interval=self.cfg.pretrain_interval,
                    auto_adjust=True,
                    progress=False,
                )
            except Exception:
                continue
            if data is None or data.empty:
                continue
            if getattr(data.columns, "nlevels", 1) > 1:
                data = data.copy()
                if "Close" in data.columns.get_level_values(0):
                    data.columns = data.columns.get_level_values(0)
                else:
                    data.columns = data.columns.get_level_values(-1)
            if "Close" not in data.columns:
                continue
            prices = data["Close"].tolist()
            if hasattr(prices, "tolist"):
                prices = prices.tolist()
            volumes = data["Volume"].tolist() if "Volume" in data else []
            opens = data["Open"].tolist() if "Open" in data else []
            highs = data["High"].tolist() if "High" in data else []
            lows = data["Low"].tolist() if "Low" in data else []
            if len(prices) < 3:
                continue
            warmup = min(self.cfg.pretrain_warmup_bars, len(prices) - 2)
            for idx in range(warmup, len(prices) - 1):
                window_prices = prices[max(0, idx - warmup) : idx + 1]
                window_volumes = volumes[max(0, idx - warmup) : idx + 1] if volumes else []
                window_opens = opens[max(0, idx - warmup) : idx + 1] if opens else []
                window_highs = highs[max(0, idx - warmup) : idx + 1] if highs else []
                window_lows = lows[max(0, idx - warmup) : idx + 1] if lows else []
                market_state = {
                    "prices": window_prices,
                    "volumes": window_volumes,
                    "opens": window_opens,
                    "highs": window_highs,
                    "lows": window_lows,
                    "last_price": window_prices[-1] if window_prices else None,
                }
                signals = []
                for name, strategy in strategies.items():
                    if strategy is None:
                        continue
                    signal = strategy.generate_signal(market_state)
                    signal["name"] = name
                    signals.append(signal)
                actions = {s.get("name"): s.get("action") for s in signals if s.get("name")}
                if not actions:
                    continue
                next_price = prices[idx + 1]
                move_pct = (next_price - prices[idx]) / prices[idx] * 100.0 if prices[idx] else 0.0
                rewards = {}
                for name, action in actions.items():
                    if action == "buy":
                        rewards[name] = move_pct
                    elif action == "sell":
                        rewards[name] = -move_pct
                    else:
                        rewards[name] = 0.0
                self._enqueue(_feature_vector(market_state), rewards)
                samples += 1
                if samples >= self.cfg.pretrain_max_samples:
                    break
            if samples >= self.cfg.pretrain_max_samples:
                break
        for _ in range(max(1, self.cfg.pretrain_epochs)):
            self._train()


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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


_FEATURE_NAMES = [
    "momentum",
    "trend",
    "volatility",
    "relative_volume",
    "session_gain_pct",
    "spread",
    "catalyst",
]


def _feature_vector(market_state: dict) -> list[float]:
    features = _extract_features(market_state) if market_state else {}
    return [float(features.get(name, 0.0) or 0.0) for name in _FEATURE_NAMES]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _last_price(market_state: dict) -> float | None:
    if not market_state:
        return None
    last_price = market_state.get("last_price")
    if last_price is not None:
        return last_price
    prices = market_state.get("prices", [])
    return prices[-1] if prices else None

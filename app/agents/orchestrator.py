# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import random
import time
from pathlib import Path
from statistics import pstdev
import logging

import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import yfinance as yf

try:
    from app.data import ai_filter as ai_filter_module
except Exception:
    ai_filter_module = None

from app.brokers.config_utils import get_alpaca_account_cfg


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
class RLOrchestratorConfig:
    enabled: bool = False
    model_type: str = "lstm"
    device: str = "auto"
    hidden_dim: int = 64
    dropout: float = 0.1
    num_layers: int = 1
    seq_len: int = 20
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
    entropy_coef: float = 0.01
    baseline_alpha: float = 0.1
    max_grad_norm: float = 1.0
    save_interval_seconds: int = 300
    score_ema_alpha: float = 0.1
    time_penalty_per_bar: float = 0.0
    pretrain_enabled: bool = True
    pretrain_in_trader: bool = False
    pretrain_provider: str = "yfinance"
    pretrain_alpaca_api_key: str = ""
    pretrain_alpaca_api_secret: str = ""
    pretrain_lookback_days: int = 30
    pretrain_interval: str = "5m"
    pretrain_window_days: int = 60
    pretrain_coverage_days: int = 365
    pretrain_step_days: int = 30
    pretrain_max_symbols: int = 20
    pretrain_max_samples: int = 2000
    pretrain_epochs: int = 2
    pretrain_warmup_bars: int = 50
    pretrain_symbols_source: str = "data"
    yf_timeout_seconds: int = 15
    yf_retries: int = 2

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
        self._reward_ema: float | None = None
        self._last_selection: dict[str, str] = {}
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
        if self.cfg.mode == "direct":
            selected = []
            for name in sorted_names:
                if scores.get(name, 0.0) < self.cfg.min_score:
                    continue
                selected = [name]
                break
            if not selected:
                selected = [sorted_names[0]]
            return selected, {selected[0]: 1.0}
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


class RLStrategyOrchestrator:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        orchestrator_cfg = cfg.get("orchestrator", cfg)
        rl_cfg = orchestrator_cfg.get("rl", orchestrator_cfg.get("ml", {}))
        self._mode = str(orchestrator_cfg.get("mode", "select"))
        self._top_k = int(orchestrator_cfg.get("top_k", 2))
        self._min_score = float(orchestrator_cfg.get("min_score", 0.0))
        self.cfg = RLOrchestratorConfig(
            enabled=bool(rl_cfg.get("enabled", False)),
            model_type=str(rl_cfg.get("model_type", "lstm")),
            device=str(rl_cfg.get("device", "auto")),
            hidden_dim=int(rl_cfg.get("hidden_dim", 64)),
            dropout=float(rl_cfg.get("dropout", 0.1)),
            num_layers=int(rl_cfg.get("num_layers", 1)),
            seq_len=int(rl_cfg.get("seq_len", 20)),
            model_path=str(rl_cfg.get("model_path", "/data/orchestrator_model.pt")),
            best_model_path=str(rl_cfg.get("best_model_path", "/data/orchestrator_model_best.pt")),
            use_best_model=bool(rl_cfg.get("use_best_model", True)),
            best_score_path=str(rl_cfg.get("best_score_path", "/data/orchestrator_model_best_score.json")),
            learning_rate=float(rl_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(rl_cfg.get("weight_decay", 1e-4)),
            batch_size=int(rl_cfg.get("batch_size", 64)),
            buffer_size=int(rl_cfg.get("buffer_size", 5000)),
            update_steps_per_bar=int(rl_cfg.get("update_steps_per_bar", 1)),
            epsilon=float(rl_cfg.get("epsilon", 0.05)),
            min_price_move_pct=float(rl_cfg.get("min_price_move_pct", 0.02)),
            reward_scale=float(rl_cfg.get("reward_scale", 1.0)),
            entropy_coef=float(rl_cfg.get("entropy_coef", 0.01)),
            baseline_alpha=float(rl_cfg.get("baseline_alpha", 0.1)),
            max_grad_norm=float(rl_cfg.get("max_grad_norm", 1.0)),
            save_interval_seconds=int(rl_cfg.get("save_interval_seconds", 300)),
            score_ema_alpha=float(rl_cfg.get("score_ema_alpha", 0.1)),
            time_penalty_per_bar=float(rl_cfg.get("time_penalty_per_bar", 0.0)),
            pretrain_enabled=bool(rl_cfg.get("pretrain", {}).get("enabled", True)),
            pretrain_in_trader=bool(rl_cfg.get("pretrain", {}).get("in_trader", False)),
            pretrain_provider=str(rl_cfg.get("pretrain", {}).get("provider", "yfinance")),
            pretrain_alpaca_api_key=str(rl_cfg.get("pretrain", {}).get("alpaca_api_key", "")),
            pretrain_alpaca_api_secret=str(rl_cfg.get("pretrain", {}).get("alpaca_api_secret", "")),
            pretrain_lookback_days=int(rl_cfg.get("pretrain", {}).get("lookback_days", 30)),
            pretrain_interval=str(rl_cfg.get("pretrain", {}).get("interval", "5m")),
            pretrain_window_days=int(rl_cfg.get("pretrain", {}).get("window_days", 60)),
            pretrain_coverage_days=int(rl_cfg.get("pretrain", {}).get("coverage_days", 365)),
            pretrain_step_days=int(rl_cfg.get("pretrain", {}).get("step_days", 30)),
            pretrain_max_symbols=int(rl_cfg.get("pretrain", {}).get("max_symbols", 20)),
            pretrain_max_samples=int(rl_cfg.get("pretrain", {}).get("max_samples", 2000)),
            pretrain_epochs=int(rl_cfg.get("pretrain", {}).get("epochs", 2)),
            pretrain_warmup_bars=int(rl_cfg.get("pretrain", {}).get("warmup_bars", 50)),
            pretrain_symbols_source=str(rl_cfg.get("pretrain", {}).get("symbols_source", "data")),
            yf_timeout_seconds=int(rl_cfg.get("pretrain", {}).get("timeout_seconds", 15)),
            yf_retries=int(rl_cfg.get("pretrain", {}).get("retries", 2)),
        )
        self._device = _resolve_device(self.cfg.device)
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self._strategy_names: list[str] = []
        self._ai_filter_cfg = cfg.get("data", {}).get("dynamic_symbols", {}).get("ai_filter", {})
        self._alpaca_cfg = get_alpaca_account_cfg(cfg)
        self._buffer: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=self.cfg.buffer_size)
        self._last_state: dict[str, dict[str, object]] = {}
        self._feature_history: dict[str, deque[list[float]]] = {}
        self._last_saved_at: float | None = None
        self._score_ema: float | None = None
        self._best_score: float | None = None
        self._reward_ema: float | None = None
        self._last_selection: dict[str, str] = {}
        self._order_feedback: dict[str, dict[str, float]] = {}

    def is_enabled(self) -> bool:
        return self.cfg.enabled

    def bootstrap(self, strategy_names: list[str], build_strategy, strategy_params: dict, data_cfg: dict) -> None:
        if not self.cfg.enabled:
            return
        self._ensure_model(strategy_names)
        self._load_best_score()
        if self._load_model():
            return
        if not self.cfg.pretrain_enabled or not self.cfg.pretrain_in_trader:
            return
        self._pretrain(strategy_names, build_strategy, strategy_params, data_cfg)
        self._save_model(self.cfg.model_path)
        if not Path(self.cfg.best_model_path).exists():
            self._save_model(self.cfg.best_model_path)

    def run_pretrain(self, strategy_names: list[str], build_strategy, strategy_params: dict, data_cfg: dict) -> None:
        if not self.cfg.enabled or not self.cfg.pretrain_enabled:
            return
        self._ensure_model(strategy_names)
        self._pretrain(strategy_names, build_strategy, strategy_params, data_cfg)
        self._save_model(self.cfg.model_path)
        self._save_model(self.cfg.best_model_path)

    def select(
        self,
        symbol: str,
        strategy_names: list[str],
        market_state: dict,
        signals: list[dict] | None = None,
    ) -> tuple[list[str], dict[str, float]]:
        if not self.cfg.enabled or not strategy_names:
            return strategy_names, {name: 1.0 for name in strategy_names}
        self._ensure_model(strategy_names)
        features = self._state_features(symbol, market_state, signals)
        sequence = self._update_sequence(symbol, features)
        probs = self._predict(sequence)
        if random.random() < self.cfg.epsilon:
            shuffled = strategy_names[:]
            random.shuffle(shuffled)
            selected = shuffled[: max(1, min(len(shuffled), self._top_k))]
            self._last_selection[symbol] = selected[0]
            return selected, {name: 1.0 for name in selected}

        ranked = sorted(strategy_names, key=lambda n: probs.get(n, 0.0), reverse=True)
        if not ranked:
            return strategy_names, {name: 1.0 for name in strategy_names}
        if len(ranked) == 1:
            self._last_selection[symbol] = ranked[0]
            return ranked, {ranked[0]: 1.0}

        selected = [name for name in ranked if probs.get(name, 0.0) >= self._min_score]
        if not selected:
            selected = ranked
        top_k = max(1, min(len(selected), self._top_k))
        selected = selected[:top_k]
        if self._mode == "direct":
            selected = [name for name in ranked if probs.get(name, 0.0) >= self._min_score]
            if not selected:
                selected = [ranked[0]]
            self._last_selection[symbol] = selected[0]
            return [selected[0]], {selected[0]: 1.0}
        if self._mode == "weight":
            weights = {name: max(probs.get(name, 0.0), 0.0) for name in selected}
            if not any(weight > 0 for weight in weights.values()):
                weights = {name: 1.0 for name in selected}
            self._last_selection[symbol] = selected[0]
            return selected, weights
        self._last_selection[symbol] = selected[0]
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
        selected = self._last_selection.get(symbol)
        if not selected or selected not in actions:
            return
        features = self._state_features(symbol, market_state, signals)
        sequence = self._update_sequence(symbol, features)
        self._last_state[symbol] = {
            "features": sequence,
            "action": str(actions.get(selected, "hold")),
            "action_idx": self._strategy_names.index(selected),
            "price": last_price,
        }

    def update(self, symbol: str, market_state: dict) -> None:
        if not self.cfg.enabled:
            return
        state = self._last_state.get(symbol)
        if not state:
            return
        prev_price = state.get("price")
        if prev_price is None or prev_price <= 0:
            return
        current_price = _last_price(market_state)
        if current_price is None or current_price <= 0:
            return
        move_pct = (current_price - prev_price) / prev_price * 100.0
        if abs(move_pct) < self.cfg.min_price_move_pct:
            self._last_state.pop(symbol, None)
            return
        action = state.get("action")
        if not action:
            self._last_state.pop(symbol, None)
            return
        if action == "buy":
            reward = move_pct - self.cfg.time_penalty_per_bar
        elif action == "sell":
            reward = -move_pct - self.cfg.time_penalty_per_bar
        else:
            reward = -self.cfg.time_penalty_per_bar
        reward *= self.cfg.reward_scale
        action_idx = state.get("action_idx")
        if action_idx is None:
            self._last_state.pop(symbol, None)
            return
        self._enqueue(state.get("features"), int(action_idx), float(reward))
        self._train()
        self._update_score(float(reward))
        self._maybe_save()
        self._maybe_save_best()
        self._last_state.pop(symbol, None)

    def on_order_update(self, response: dict) -> None:
        symbol = response.get("symbol")
        if not symbol:
            return
        self._order_feedback[symbol] = _order_feedback_features(response)

    def _ensure_model(self, strategy_names: list[str]) -> None:
        if self._model is not None:
            return
        self._strategy_names = strategy_names
        self._input_dim = len(_feature_vector({})) + _ai_feature_dim() + len(strategy_names) * 2 + _order_feature_dim()
        self._output_dim = len(strategy_names)
        if self.cfg.model_type == "lstm":
            self._model = _LSTMModel(
                self._input_dim,
                self._output_dim,
                self.cfg.hidden_dim,
                self.cfg.num_layers,
                self.cfg.dropout,
            ).to(self._device)
        else:
            self._model = _MLP(self._input_dim, self._output_dim, self.cfg.hidden_dim, self.cfg.dropout).to(
                self._device
            )
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def _predict(self, features: list[float] | list[list[float]]) -> dict[str, float]:
        if not self._model:
            return {name: 0.0 for name in self._strategy_names}
        features = self._normalize_features(features)
        vec = torch.tensor(features, dtype=torch.float32, device=self._device)
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        if vec.dim() == 2 and self.cfg.model_type == "lstm":
            vec = vec.unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(vec).squeeze(0)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        return {name: float(prob) for name, prob in zip(self._strategy_names, probs)}

    def _enqueue(self, features: list[float] | list[list[float]] | None, action_idx: int, reward: float) -> None:
        if features is None:
            return
        features = self._normalize_features(features)
        x = torch.tensor(features, dtype=torch.float32)
        action = torch.tensor(int(action_idx), dtype=torch.long)
        reward_tensor = torch.tensor(float(reward), dtype=torch.float32)
        self._buffer.append((x, action, reward_tensor))

    def _train(self) -> None:
        if not self._model or not self._optimizer:
            return
        if len(self._buffer) < self.cfg.batch_size:
            return
        self._model.train()
        for _ in range(max(1, self.cfg.update_steps_per_bar)):
            batch = random.sample(list(self._buffer), self.cfg.batch_size)
            x = torch.stack([item[0] for item in batch]).to(self._device)
            actions = torch.stack([item[1] for item in batch]).to(self._device)
            rewards = torch.stack([item[2] for item in batch]).to(self._device)
            if self.cfg.model_type == "lstm":
                x = x.unsqueeze(0) if x.dim() == 2 else x
            logits = self._model(x)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            reward_mean = float(rewards.mean().item())
            if self._reward_ema is None:
                self._reward_ema = reward_mean
            else:
                alpha = max(0.0, min(self.cfg.baseline_alpha, 1.0))
                self._reward_ema = (1.0 - alpha) * self._reward_ema + alpha * reward_mean
            baseline = torch.tensor(self._reward_ema or 0.0, device=self._device)
            advantages = rewards - baseline
            loss = -(log_probs * advantages).mean() - self.cfg.entropy_coef * entropy.mean()
            self._optimizer.zero_grad()
            loss.backward()
            if self.cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.cfg.max_grad_norm)
            self._optimizer.step()

    def _update_score(self, reward: float) -> None:
        alpha = self.cfg.score_ema_alpha
        if self._score_ema is None:
            self._score_ema = reward
        else:
            self._score_ema = alpha * reward + (1.0 - alpha) * self._score_ema

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

    def _normalize_features(self, features: list[float] | list[list[float]]) -> list[float] | list[list[float]]:
        if self.cfg.model_type != "mlp":
            return features
        if not features or not isinstance(features, list):
            return features
        if isinstance(features[0], list):
            return features[-1]
        return features

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
        except (OSError, RuntimeError):
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
            for data in _iter_pretrain_windows(symbol, self.cfg):
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
                seq = deque(maxlen=self.cfg.seq_len)
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
                    features = self._state_features(symbol, market_state, signals)
                    seq.append(features)
                    sequence = list(seq)
                    if len(sequence) < self.cfg.seq_len:
                        sequence = _pad_sequence(sequence, self.cfg.seq_len, self._input_dim or len(sequence[-1]))
                    actions = {s.get("name"): s.get("action") for s in signals if s.get("name")}
                    if not actions:
                        continue
                    next_price = prices[idx + 1]
                    move_pct = (next_price - prices[idx]) / prices[idx] * 100.0 if prices[idx] else 0.0
                    rewards = {}
                    for name, action in actions.items():
                        if action == "buy":
                            rewards[name] = move_pct - self.cfg.time_penalty_per_bar
                        elif action == "sell":
                            rewards[name] = -move_pct - self.cfg.time_penalty_per_bar
                        else:
                            rewards[name] = -self.cfg.time_penalty_per_bar
                    best_name = max(rewards, key=rewards.get) if rewards else None
                    if not best_name:
                        continue
                    reward = rewards[best_name] * self.cfg.reward_scale
                    train_features = sequence if self.cfg.model_type == "lstm" else features
                    self._enqueue(train_features, self._strategy_names.index(best_name), float(reward))
                    samples += 1
                    if samples >= self.cfg.pretrain_max_samples:
                        break
                if samples >= self.cfg.pretrain_max_samples:
                    break
            if samples >= self.cfg.pretrain_max_samples:
                break
        for _ in range(max(1, self.cfg.pretrain_epochs)):
            self._train()

    def _update_sequence(self, symbol: str, features: list[float]) -> list[list[float]]:
        history = self._feature_history.get(symbol)
        if history is None:
            history = deque(maxlen=self.cfg.seq_len)
            self._feature_history[symbol] = history
        history.append(features)
        sequence = list(history)
        if len(sequence) < self.cfg.seq_len:
            sequence = _pad_sequence(sequence, self.cfg.seq_len, len(features))
        return sequence

    def _state_features(self, symbol: str, market_state: dict, signals: list[dict] | None) -> list[float]:
        base = _feature_vector(market_state)
        actions = signals or []
        ai_features = self._ai_features(symbol, market_state, actions)
        signal_features = _signal_feature_vector(self._strategy_names, actions)
        order_features = _order_feedback_vector(self._order_feedback.get(symbol))
        return base + ai_features + signal_features + order_features

    def _ai_features(self, symbol: str, market_state: dict, signals: list[dict]) -> list[float]:
        if ai_filter_module is None or not self._ai_filter_cfg:
            return [0.0] * _ai_feature_dim()
        has_order = any(s.get("action") in {"buy", "sell", "exit"} for s in signals)
        if not has_order:
            return [0.0] * _ai_feature_dim()
        window = int(self._ai_filter_cfg.get("window", 20))
        catalyst = bool(market_state.get("catalyst", False))
        prices = market_state.get("prices") or []
        volumes = market_state.get("volumes") or []
        features = ai_filter_module.build_feature_vector_from_series(
            prices,
            volumes,
            window,
            catalyst,
            interval=str(self._ai_filter_cfg.get("interval", "5m")),
        )
        allow_fetch = bool(self._ai_filter_cfg.get("allow_orchestrator_fetch", False))
        if features is None and allow_fetch:
            api_key = str(self._alpaca_cfg.get("api_key", ""))
            api_secret = str(self._alpaca_cfg.get("api_secret", ""))
            features = ai_filter_module.latest_features_for_symbol(
                symbol,
                api_key,
                api_secret,
                self._ai_filter_cfg,
                catalyst,
            )
        if features is None:
            return [0.0] * _ai_feature_dim()
        return [float(val) for val in features.tolist()]


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


class _LSTMModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last)

def _portfolio_features(portfolio: dict | None) -> dict[str, float]:
    if not isinstance(portfolio, dict):
        return {"cash_pct": 0.0, "buying_power_pct": 0.0}
    equity = float(portfolio.get("equity") or portfolio.get("last_equity") or 0.0)
    cash = float(portfolio.get("cash") or 0.0)
    buying_power = float(portfolio.get("buying_power") or 0.0)
    if equity <= 0:
        return {"cash_pct": 0.0, "buying_power_pct": 0.0}
    return {
        "cash_pct": cash / equity,
        "buying_power_pct": buying_power / equity,
    }


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
    portfolio_features = _portfolio_features(market_state.get("portfolio"))

    return {
        "momentum": momentum,
        "trend": trend,
        "volatility": volatility,
        "relative_volume": float(market_state.get("relative_volume", 0.0) or 0.0),
        "session_gain_pct": float(market_state.get("session_gain_pct", 0.0) or 0.0),
        "spread": float(market_state.get("spread_pct", 0.0) or 0.0),
        "catalyst": 1.0 if market_state.get("catalyst") else 0.0,
        "signal_30m_return_pct": float(market_state.get("signal_30m_return_pct", 0.0) or 0.0),
        "signal_60m_return_pct": float(market_state.get("signal_60m_return_pct", 0.0) or 0.0),
        "signal_early_volume_pct": float(market_state.get("signal_early_volume_pct", 0.0) or 0.0),
        "signal_runup_pct": float(market_state.get("signal_runup_pct", 0.0) or 0.0),
        "signal_drawdown_pct": float(market_state.get("signal_drawdown_pct", 0.0) or 0.0),
        "signal_abs_move": float(market_state.get("signal_abs_move", 0.0) or 0.0),
        "signal_runup_abs": float(market_state.get("signal_runup_abs", 0.0) or 0.0),
        "signal_drawdown_abs": float(market_state.get("signal_drawdown_abs", 0.0) or 0.0),
        "cash_pct": float(portfolio_features.get("cash_pct", 0.0) or 0.0),
        "buying_power_pct": float(portfolio_features.get("buying_power_pct", 0.0) or 0.0),
    }


_FEATURE_NAMES = [
    "momentum",
    "trend",
    "volatility",
    "relative_volume",
    "session_gain_pct",
    "spread",
    "catalyst",
    "signal_30m_return_pct",
    "signal_60m_return_pct",
    "signal_early_volume_pct",
    "signal_runup_pct",
    "signal_drawdown_pct",
    "signal_abs_move",
    "signal_runup_abs",
    "signal_drawdown_abs",
    "cash_pct",
    "buying_power_pct",
]


def _feature_vector(market_state: dict) -> list[float]:
    features = _extract_features(market_state) if market_state else {}
    return [float(features.get(name, 0.0) or 0.0) for name in _FEATURE_NAMES]


def _ai_feature_dim() -> int:
    return 14


def _order_feature_dim() -> int:
    return 4


def _signal_feature_vector(strategy_names: list[str], signals: list[dict]) -> list[float]:
    action_map = {}
    reduce_map = {}
    for signal in signals:
        name = signal.get("name")
        if not name:
            continue
        action_map[name] = signal.get("action", "hold")
        reduce_map[name] = float(signal.get("reduce_pct", 1.0))
    features = []
    for name in strategy_names:
        action = str(action_map.get(name, "hold")).lower()
        if action == "buy":
            action_val = 1.0
        elif action == "sell":
            action_val = -1.0
        elif action == "exit":
            action_val = -0.5
        else:
            action_val = 0.0
        features.append(action_val)
        features.append(float(reduce_map.get(name, 0.0)))
    return features


def _order_feedback_features(response: dict) -> dict[str, float]:
    status = str(response.get("status", "")).lower()
    if status in {"filled", "completed", "done"}:
        status_val = 1.0
    elif status in {"canceled", "cancelled"}:
        status_val = -1.0
    elif status in {"rejected", "failed"}:
        status_val = -0.5
    elif status in {"submitted", "accepted"}:
        status_val = 0.5
    else:
        status_val = 0.0
    side = str(response.get("side", "")).lower()
    side_val = 1.0 if side == "buy" else (-1.0 if side == "sell" else 0.0)
    qty = float(response.get("qty") or 0.0)
    filled_qty = float(response.get("filled_qty") or 0.0)
    fill_ratio = filled_qty / qty if qty else 0.0
    broker = str(response.get("broker", "")).lower()
    broker_val = 1.0 if broker == "alpaca" else (2.0 if broker == "ibkr" else 0.0)
    return {
        "order_status": status_val,
        "order_side": side_val,
        "order_fill_ratio": fill_ratio,
        "order_broker": broker_val,
    }


def _order_feedback_vector(state: dict[str, float] | None) -> list[float]:
    state = state or {}
    return [
        float(state.get("order_status", 0.0)),
        float(state.get("order_side", 0.0)),
        float(state.get("order_fill_ratio", 0.0)),
        float(state.get("order_broker", 0.0)),
    ]


def _resolve_device(device: str) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if device != "auto":
        return device
    return "cpu"


def _last_price(market_state: dict) -> float | None:
    if not market_state:
        return None
    last_price = market_state.get("last_price")
    if last_price is not None:
        return last_price
    prices = market_state.get("prices", [])
    return prices[-1] if prices else None


def _pad_sequence(sequence: list[list[float]], target_len: int, feature_dim: int) -> list[list[float]]:
    if len(sequence) >= target_len:
        return sequence
    pad = [0.0] * feature_dim
    needed = target_len - len(sequence)
    return [pad for _ in range(needed)] + sequence


def _download_yf(
    symbol: str,
    lookback_days: int,
    interval: str,
    timeout_seconds: int,
    retries: int,
):
    for attempt in range(retries + 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                yf.download,
                tickers=symbol,
                period=f"{lookback_days}d",
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError:
                if attempt >= retries:
                    return None
            except Exception:
                if attempt >= retries:
                    return None
    return None


def _iter_pretrain_windows(symbol: str, cfg: RLOrchestratorConfig):
    provider = cfg.pretrain_provider
    if provider == "alpaca":
        yield from _iter_alpaca_pretrain_windows(symbol, cfg)
        return
    if cfg.pretrain_interval != "5m":
        data = _download_yf(
            symbol,
            lookback_days=cfg.pretrain_lookback_days,
            interval=cfg.pretrain_interval,
            timeout_seconds=cfg.yf_timeout_seconds,
            retries=cfg.yf_retries,
        )
        if data is not None:
            yield data
        return

    now = datetime.now(timezone.utc)
    end = now
    coverage_days = max(cfg.pretrain_coverage_days, cfg.pretrain_window_days)
    if str(cfg.pretrain_interval).endswith("m"):
        coverage_days = min(coverage_days, 59)
    earliest = now - timedelta(days=coverage_days)
    step = max(1, cfg.pretrain_step_days)
    window = max(1, cfg.pretrain_window_days)
    while end > earliest:
        start = end - timedelta(days=window)
        data = _download_yf_range(
            symbol,
            start=start,
            end=end,
            interval=cfg.pretrain_interval,
            timeout_seconds=cfg.yf_timeout_seconds,
            retries=cfg.yf_retries,
        )
        if data is not None:
            yield data
        end = end - timedelta(days=step)


def _iter_alpaca_pretrain_windows(symbol: str, cfg: RLOrchestratorConfig):
    api_key = cfg.pretrain_alpaca_api_key
    api_secret = cfg.pretrain_alpaca_api_secret
    if not api_key or not api_secret:
        return
    now = datetime.now(timezone.utc)
    end = now
    coverage_days = max(cfg.pretrain_coverage_days, cfg.pretrain_window_days)
    earliest = now - timedelta(days=coverage_days)
    step = max(1, cfg.pretrain_step_days)
    window = max(1, cfg.pretrain_window_days)
    while end > earliest:
        start = end - timedelta(days=window)
        data = _download_alpaca_bars_range(
            symbol,
            start=start,
            end=end,
            interval=cfg.pretrain_interval,
            api_key=api_key,
            api_secret=api_secret,
        )
        if data is not None:
            yield data
        end = end - timedelta(days=step)


def _download_yf_range(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    timeout_seconds: int,
    retries: int,
):
    for attempt in range(retries + 1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                yf.download,
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError:
                if attempt >= retries:
                    return None
            except Exception:
                if attempt >= retries:
                    return None
    return None


def _download_alpaca_bars_range(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
    api_key: str,
    api_secret: str,
):
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception:
        return None

    if interval.endswith("m"):
        timeframe = TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)
    elif interval.endswith("h"):
        timeframe = TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
    elif interval.endswith("d"):
        timeframe = TimeFrame(int(interval[:-1]), TimeFrameUnit.Day)
    else:
        timeframe = TimeFrame(1, TimeFrameUnit.Day)

    client = StockHistoricalDataClient(api_key, api_secret)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment="raw",
    )
    try:
        data = client.get_stock_bars(req).df
    except Exception:
        return None
    if data is None or data.empty:
        return None
    if hasattr(data.index, "levels"):
        data = data.copy()
        data.index = data.index.get_level_values(-1)
    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return data

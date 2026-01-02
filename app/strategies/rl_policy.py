# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from stable_baselines3 import PPO

from app.learning.features import build_observation
from app.strategies.base import Strategy


class RLPolicyStrategy(Strategy):
    def __init__(
        self,
        model_path: str,
        window_size: int = 50,
        device: str = "auto",
        feature_config: dict | None = None,
    ):
        self.window_size = window_size
        self.device = _resolve_device(device)
        self.model_path = model_path
        self.feature_config = feature_config or {}
        self.model = None
        self.position = 0.0
        self.prices = deque(maxlen=window_size * 4)
        self.volumes = deque(maxlen=window_size * 4)
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"RL model not found at {model_path}")
        self.model = PPO.load(str(path), device=self.device)
        logging.info("Loaded RL model from %s", model_path)

    def _update_state(self, market_state: dict) -> None:
        prices = market_state.get("prices", [])
        volumes = market_state.get("volumes", [])
        if prices:
            self.prices = deque((float(p) for p in prices), maxlen=self.prices.maxlen)
        if volumes:
            self.volumes = deque((float(v) for v in volumes), maxlen=self.volumes.maxlen)
        if self.prices and not self.volumes:
            self.volumes = deque([1.0] * len(self.prices), maxlen=self.volumes.maxlen)
        if len(self.volumes) < len(self.prices):
            self.volumes.extend([self.volumes[-1]] * (len(self.prices) - len(self.volumes)))

    def generate_signal(self, market_state: dict) -> dict:
        self._update_state(market_state)
        if len(self.prices) < 2:
            return {"action": "hold"}

        closes = list(self.prices)
        volumes = list(self.volumes)
        obs = build_observation(
            closes=closes,
            volumes=volumes,
            window_size=self.window_size,
            position=self.position,
            cash_pct=1.0,
            feature_config=self.feature_config,
        )
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        if action == 1:
            self.position = 1.0
            return {"action": "buy"}
        if action == 2:
            self.position = -1.0
            return {"action": "sell"}
        return {"action": "hold"}


def _resolve_device(device: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if device != "auto":
        return device
    return "cpu"

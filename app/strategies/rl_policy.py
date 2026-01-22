# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING
from pathlib import Path

from app.learning.features import build_observation, observation_size, risk_feature_vector
from app.strategies.base import Strategy
from app.utils.gpu_state import is_gpu_disabled

if TYPE_CHECKING:
    from app.learning.drift import DriftMonitor


class RLPolicyStrategy(Strategy):
    def __init__(
        self,
        model_path: str,
        window_size: int = 50,
        device: str = "auto",
        feature_config: dict | None = None,
        drift_monitor: "DriftMonitor | None" = None,
        include_features: bool = False,
        risk_cfg: dict | None = None,
    ):
        self.window_size = window_size
        self.device = _resolve_device(device)
        self.model_path = model_path
        self.feature_config = feature_config or {}
        self._risk_cfg = risk_cfg or {}
        self._drift_monitor = drift_monitor
        self._include_features = include_features
        self.model = None
        self.position = 0.0
        self.prices = deque(maxlen=window_size * 4)
        self.volumes = deque(maxlen=window_size * 4)
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"RL model not found at {model_path}")
        ppo_cls = _load_ppo()
        model = ppo_cls.load(str(path), device=self.device, custom_objects=_sb3_custom_objects())
        expected = observation_size(self.window_size, self.feature_config)
        actual = 0
        try:
            shape = getattr(model.observation_space, "shape", None)
            if shape:
                actual = int(shape[0] or 0)
        except Exception:
            actual = 0
        if actual and expected and actual != expected:
            raise ValueError(f"RL model observation size mismatch: {actual} != {expected}")
        self.model = model
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
        portfolio = market_state.get("portfolio", {}) if isinstance(market_state, dict) else {}
        cash_pct = float(portfolio.get("cash_pct", 1.0) or 1.0)
        buying_power_pct = float(portfolio.get("buying_power_pct", cash_pct) or cash_pct)
        risk_outcome = market_state.get("risk_outcome") if isinstance(market_state, dict) else None
        risk_features = risk_feature_vector(self._risk_cfg, risk_outcome, include_decision=True)
        obs = build_observation(
            closes=closes,
            volumes=volumes,
            window_size=self.window_size,
            position=self.position,
            cash_pct=cash_pct,
            buying_power_pct=buying_power_pct,
            feature_config=self.feature_config,
            risk_features=risk_features,
        )
        if self._drift_monitor:
            self._drift_monitor.update_features(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        signal = {"action": "hold"}
        if action == 1:
            self.position = 1.0
            signal["action"] = "buy"
        elif action == 2:
            self.position = -1.0
            signal["action"] = "sell"
        if self._include_features:
            signal["features"] = obs.tolist()
        return signal


def _resolve_device(device: str) -> str:
    if is_gpu_disabled():
        if device != "cpu":
            logging.warning("GPU disabled; forcing CPU for RL policy.")
        return "cpu"
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_ppo():
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise ImportError(
            "stable_baselines3 is required for RL policy models; install the RL extras."
        ) from exc
    return PPO


def _sb3_custom_objects() -> dict:
    return {
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 0.0,
    }

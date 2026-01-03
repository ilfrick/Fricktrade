# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionContext:
    symbol: str
    market_state: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)
    broker: str | None = None
    signals: list[dict[str, Any]] = field(default_factory=list)
    action: str | None = None
    reason: str | None = None
    stages: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "broker": self.broker,
            "action": self.action,
            "reason": self.reason,
            "signals": list(self.signals),
            "stages": dict(self.stages),
            "metadata": dict(self.metadata),
        }

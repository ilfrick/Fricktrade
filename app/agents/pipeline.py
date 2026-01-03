# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from app.agents.decision_context import DecisionContext


class DecisionPipeline:
    def __init__(self, agent):
        self._agent = agent

    def run(self, symbol: str, market_state: dict):
        _ = DecisionContext(symbol=symbol, market_state=market_state, portfolio=market_state.get("portfolio", {}))
        return self._agent.run_once(symbol, market_state)

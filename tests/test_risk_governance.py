# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import pytest

pytest.importorskip("prometheus_client")

from app.agents.trader import _group_exposure, _realized_volatility_pct, _var_cvar_from_history


def test_var_cvar_from_history_returns_defaults():
    stats = _var_cvar_from_history([100.0], 0.95)
    assert stats["var_pct"] == 0.0
    assert stats["cvar_pct"] == 0.0


def test_var_cvar_from_history_computes_tail_risk():
    history = [100.0, 102.0, 99.0, 95.0, 96.0]
    stats = _var_cvar_from_history(history, 0.95)
    assert stats["var_pct"] >= 0.0
    assert stats["cvar_pct"] >= stats["var_pct"]


def test_group_exposure_aggregates_by_bucket():
    portfolio = {
        "positions": {
            "AAPL": {"value": 5000.0},
            "MSFT": {"value": 3000.0},
            "ISP": {"value": -2000.0},
        }
    }

    def mapper(symbol: str) -> str:
        return {"AAPL": "US", "MSFT": "US", "ISP": "EU"}.get(symbol)

    exposure = _group_exposure(portfolio, mapper)
    assert exposure["US"] == 8000.0
    assert exposure["EU"] == 2000.0


def test_realized_volatility_pct_handles_flat_series():
    market_state = {"prices": [100.0, 100.0, 100.0, 100.0]}
    assert _realized_volatility_pct(market_state) == 0.0

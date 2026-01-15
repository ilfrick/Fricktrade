# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json

import pytest

pytest.importorskip("prometheus_client")
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.backtest.sampling import build_windows, sample_universe
from app.backtest.agent_engine import _backtest_cfg_override, _load_backtest_news, _apply_news_cache, SimBroker
from app.agents.trader import TradingAgent


def _write_csv(path: Path, closes: list[float], volumes: list[int]) -> None:
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range("2025-01-01", periods=len(closes), freq="D"),
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": volumes,
        }
    )
    df.to_csv(path, index=False)


def test_build_windows_splits_range() -> None:
    start = datetime(2025, 1, 1)
    end = datetime(2025, 2, 1)
    windows = build_windows(start, end, window_days=10, step_days=5)
    assert windows
    assert windows[0][0] == start
    assert windows[-1][1] <= end


def test_sample_universe_tiers(tmp_path: Path) -> None:
    interval = "1d"
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for idx, symbol in enumerate(symbols):
        closes = [10 + idx] * 5
        volumes = [100 * (idx + 1)] * 5
        _write_csv(tmp_path / f"{symbol}_{interval}.csv", closes, volumes)
    selected = sample_universe(tmp_path, interval, sample_per_tier=1, tiers=3, seed=42)
    assert len(selected) == 3
    assert all(isinstance(sym, str) for sym in selected)


def test_backtest_cfg_override_disables_local_news() -> None:
    cfg = {
        "data": {"dynamic_symbols": {}},
        "news": {"enabled": True},
        "backtest": {"news_source": "local", "news_enabled": True},
        "orchestrator": {},
        "execution": {},
    }
    result = _backtest_cfg_override(cfg)
    assert result["news"]["enabled"] is False


def test_load_and_apply_backtest_news(tmp_path: Path) -> None:
    news_path = tmp_path / "news.json"
    payload = {"2025-01-02": ["AAA", "BBB"]}
    news_path.write_text(json.dumps(payload), encoding="utf-8")
    news_cfg = {"news_source": "local", "news_path": str(news_path)}
    cache = _load_backtest_news(news_cfg)
    assert cache["2025-01-02"] == {"AAA", "BBB"}

    broker = SimBroker(1000, 0.0)
    agent = TradingAgent(
        broker,
        {
            "risk": {
                "max_daily_loss_pct": 100.0,
                "max_position_size_pct": 100.0,
                "max_short_exposure_pct": 100.0,
                "max_portfolio_leverage": 100.0,
                "circuit_breaker_drawdown_pct": 100.0,
            },
            "strategy": {"params": {"lookback_minutes": 10}, "name": "intraday_momentum"},
            "data": {},
            "execution": {},
        },
    )
    ts = datetime(2025, 1, 2)
    _apply_news_cache(agent, cache, ts)
    assert agent._news_cache.get("AAA") is True

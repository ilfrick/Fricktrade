# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import pandas as pd

from app.data import ai_filter


def test_fetch_bars_single_symbol_mapping(monkeypatch):
    df = pd.DataFrame({"close": [1.0, 2.0], "volume": [10.0, 12.0]})

    def fake_fetch(client, request, timeout, retries):
        return df

    monkeypatch.setattr(ai_filter, "_fetch_with_retries", fake_fetch)
    cfg = _cfg()
    bars = ai_filter._fetch_bars(["AAA"], "key", "secret", cfg, limit_symbols=None)
    assert "AAA" in bars
    assert bars["AAA"].equals(df)


def test_fetch_bars_symbol_column_mapping(monkeypatch):
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
        }
    )

    def fake_fetch(client, request, timeout, retries):
        return df

    monkeypatch.setattr(ai_filter, "_fetch_with_retries", fake_fetch)
    cfg = _cfg()
    bars = ai_filter._fetch_bars(["AAA", "BBB"], "key", "secret", cfg, limit_symbols=None)
    assert set(bars.keys()) == {"AAA", "BBB"}
    assert list(bars["AAA"]["close"]) == [1.0]
    assert list(bars["BBB"]["close"]) == [2.0]


def _cfg() -> ai_filter.AISymbolFilterConfig:
    return ai_filter.AISymbolFilterConfig(
        interval="1m",
        lookback_days=1,
        window=2,
        retrain_hours=1,
        model_path="/tmp/ai_filter.pt",
        train_max_symbols=10,
        max_samples_per_symbol=10,
        online_enabled=False,
        online_learning_rate=0.01,
        online_steps=1,
        online_max_symbols=10,
        news_enabled=False,
        news_lookback_hours=1,
        news_keywords=[],
        news_timeout_seconds=1,
        news_retries=1,
        news_base_url="https://data.alpaca.markets",
        news_provider="alpaca",
        timeout_seconds=1,
        retries=0,
        objective="return",
        time_penalty_per_bar=0.0,
        feed="iex",
    )

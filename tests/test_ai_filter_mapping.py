# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import pandas as pd
import pytest

pytest.importorskip("gymnasium")

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
        model_path="/tmp/ai_filter.zip",
        model_type="ppo",
        train_max_symbols=10,
        max_samples_per_symbol=10,
        online_enabled=False,
        online_learning_rate=0.01,
        online_steps=1,
        online_timesteps=1,
        online_max_symbols=10,
        rl_timesteps=10,
        rl_learning_rate=0.0003,
        rl_batch_size=8,
        rl_n_steps=8,
        rl_gamma=0.99,
        rl_ent_coef=0.01,
        rl_clip_range=0.2,
        rl_gae_lambda=0.95,
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
        provider="alpaca",
        # Add the missing arguments with default or sensible test values
        market_cache_enabled=False,
        market_cache_redis_url="redis://localhost:6379/0", # Using localhost for test
        market_cache_file_dir="/tmp/market_cache",
        market_cache_cache_only=False,
        market_cache_ignore_staleness=False,
        market_cache_allow_pickle=False,
        market_cache_max_age_multiplier=1,
        keras_enabled=False,
        keras_model_path="/tmp/keras_model.keras",
        keras_interval="5m",
        keras_weight=0.0,
        keras_score_mode="expected_return",
        device="cpu", # Default to CPU for tests
        return_ranker_enabled=False,
        return_ranker_model_path="/tmp/return_ranker.zip",
        return_ranker_data_dir="/tmp/return_ranker_data",
        return_ranker_retrain_hours=24,
        return_ranker_max_age_days=3,
    )

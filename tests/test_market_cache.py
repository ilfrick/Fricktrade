# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

pytest.importorskip("prometheus_client")


def test_market_cache_json_round_trip(tmp_path, monkeypatch) -> None:
    import app.data.market_cache as market_cache

    monkeypatch.setattr(market_cache, "redis", None)

    cache = market_cache.MarketCache(
        "redis://unused",
        str(tmp_path),
        ignore_staleness=True,
        allow_pickle=False,
    )
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.9, 1.8],
            "Close": [1.1, 2.1],
            "Volume": [100.0, 200.0],
        },
        index=pd.to_datetime(["2025-01-01 09:30", "2025-01-01 09:31"]),
    )
    cache.set_bars({"AAA": df}, "1m", ttl_seconds=60)
    results = cache.get_bars(["AAA"], "1m", max_age_seconds=60)
    assert "AAA" in results
    pd.testing.assert_frame_equal(results["AAA"], df, check_dtype=False)
    assert (tmp_path / "bars" / "1m" / "AAA.json").exists()


def test_market_cache_pickle_blocked(tmp_path, monkeypatch) -> None:
    import app.data.market_cache as market_cache

    monkeypatch.setattr(market_cache, "redis", None)

    cache = market_cache.MarketCache(
        "redis://unused",
        str(tmp_path),
        ignore_staleness=True,
        allow_pickle=False,
    )
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.5],
            "Low": [0.9],
            "Close": [1.1],
            "Volume": [100.0],
        },
        index=pd.to_datetime(["2025-01-01 09:30"]),
    )
    bars_dir = tmp_path / "bars" / "1m"
    bars_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = bars_dir / "BBB.pkl"
    record = {"updated_at": time.time(), "data": df}
    pd.to_pickle(record, pkl_path)

    results = cache.get_bars(["BBB"], "1m", max_age_seconds=60)
    assert "BBB" not in results


def test_filtered_symbols_per_symbol_staleness(tmp_path, monkeypatch) -> None:
    import app.data.market_cache as market_cache

    monkeypatch.setattr(market_cache, "redis", None)

    cache = market_cache.MarketCache(
        "redis://unused",
        str(tmp_path),
        ignore_staleness=False,
        allow_pickle=False,
    )
    per_symbol_dir = tmp_path / "filtered" / "1m"
    per_symbol_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    (per_symbol_dir / "AAA.json").write_text(json.dumps({"updated_at": now - 120}), encoding="utf-8")
    (per_symbol_dir / "BBB.json").write_text(json.dumps({"updated_at": now - 10}), encoding="utf-8")

    symbols = cache.get_filtered_symbols("1m", max_age_seconds=60)
    assert symbols is not None
    assert set(symbols) == {"BBB"}

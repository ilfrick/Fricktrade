# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")

from app.strategies.top_movers_rf import TopMoversRFStrategy


def _make_symbol_day_frame(symbol_idx: int, day_idx: int) -> pd.DataFrame:
    bars = 120
    start = 10.0 + symbol_idx
    minutes = np.arange(bars, dtype=float)
    drift_map = [0.022, 0.006, -0.003]
    drift = drift_map[symbol_idx % len(drift_map)]
    trend = start * (1.0 + drift * (minutes / max(bars - 1, 1)))
    dip_center = 45 + day_idx % 10
    dip = 0.7 * np.exp(-((minutes - dip_center) ** 2) / 180.0)
    noise = 0.03 * np.sin(minutes / 6.0 + day_idx)
    close = np.maximum(trend - dip + noise, 0.5)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.04
    low = np.minimum(open_, close) - 0.04
    volume = 2000 + symbol_idx * 400 + (np.sin(minutes / 9.0 + symbol_idx) + 1.0) * 350
    start_ts = pd.Timestamp("2026-02-01T14:30:00Z") + pd.Timedelta(days=int(day_idx))
    ts = pd.date_range(start=start_ts, periods=bars, freq="1min")
    return pd.DataFrame(
        {
            "datetime": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _write_training_files(base_dir, days: int = 6) -> tuple[list[float], list[float], list[float], list[float]]:
    symbols = ["AAA", "BBB", "CCC"]
    sample_prices = sample_highs = sample_lows = sample_volumes = None
    for d in range(days):
        day = date(2026, 2, 1 + d)
        for idx, sym in enumerate(symbols):
            frame = _make_symbol_day_frame(idx, d)
            frame.to_csv(base_dir / f"{sym}_{day.isoformat()}_1m.csv", index=False)
            if sym == "AAA" and d == days - 1:
                sample_prices = frame["close"].tolist()
                sample_highs = frame["high"].tolist()
                sample_lows = frame["low"].tolist()
                sample_volumes = frame["volume"].tolist()
    return sample_prices, sample_highs, sample_lows, sample_volumes


def test_top_movers_rf_trains_and_emits_buy(tmp_path) -> None:
    prices, highs, lows, volumes = _write_training_files(tmp_path)
    model_path = tmp_path / "models" / "top_movers_rf.pkl"
    strat = TopMoversRFStrategy(
        {
            "data_dir": str(tmp_path),
            "model_path": str(model_path),
            "auto_train_on_start": True,
            "retrain_if_older_hours": 0,
            "min_training_days": 4,
            "min_bars": 60,
            "top_n": 1,
            "runtime_interval": "1m",
            "training_interval": "1m",
            "buy_nowcast_min": 0.0,
            "buy_entry_min": 0.0,
            "buy_score_min": 0.0,
            "min_buying_power": 1.0,
        },
        runtime_interval="1m",
    )
    assert model_path.exists()
    signal = strat.generate_signal(
        {
            "symbol": "AAA",
            "session_prices": prices,
            "session_highs": highs,
            "session_lows": lows,
            "session_volumes": volumes,
            "portfolio": {"cash": 10000.0, "buying_power": 10000.0, "positions": {}},
            "account_flags": {},
        }
    )
    assert signal["action"] == "buy"
    assert 0.0 <= float(signal.get("score", 0.0)) <= 1.0
    assert 0.0 <= float(signal.get("nowcast_score", 0.0)) <= 1.0
    assert 0.0 <= float(signal.get("entry_score", 0.0)) <= 1.0


def test_top_movers_rf_respects_account_blocks(tmp_path) -> None:
    prices, highs, lows, volumes = _write_training_files(tmp_path)
    model_path = tmp_path / "models" / "top_movers_rf.pkl"
    strat = TopMoversRFStrategy(
        {
            "data_dir": str(tmp_path),
            "model_path": str(model_path),
            "auto_train_on_start": True,
            "retrain_if_older_hours": 0,
            "min_training_days": 4,
            "min_bars": 60,
            "top_n": 1,
            "runtime_interval": "1m",
            "training_interval": "1m",
            "buy_nowcast_min": 0.0,
            "buy_entry_min": 0.0,
            "buy_score_min": 0.0,
            "min_buying_power": 1.0,
        },
        runtime_interval="1m",
    )
    signal = strat.generate_signal(
        {
            "symbol": "AAA",
            "session_prices": prices,
            "session_highs": highs,
            "session_lows": lows,
            "session_volumes": volumes,
            "portfolio": {"cash": 10000.0, "buying_power": 10000.0, "positions": {}},
            "account_flags": {"account_blocked": True},
        }
    )
    assert signal["action"] == "hold"
    assert signal.get("reason") == "account_blocked"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.backtest.splits import (
    block_bootstrap_ci, calendar_holdout_split, common_timestamp_frames,
    purge_crossing_trades,
)
from app.risk import alpha_gate
from app.risk.trials import append_trial, count_trials, deflated_sharpe_probability


def test_calendar_split_uses_same_cut_for_every_symbol():
    idx_a = pd.date_range("2025-01-01", periods=12, freq="D")
    idx_b = idx_a.delete(2)
    frames = {
        "A": pd.DataFrame({"x": range(len(idx_a))}, index=idx_a),
        "B": pd.DataFrame({"x": range(len(idx_b))}, index=idx_b),
    }
    split = calendar_holdout_split(frames, fraction=0.5)
    assert split.cutoff == split.holdout["A"].index[0] == split.holdout["B"].index[0]
    assert split.common_index.equals(common_timestamp_frames(frames)["A"].index)
    assert all(frame.index.max() < split.cutoff for frame in split.train.values())


def test_purge_removes_crossing_trades_and_embargo():
    cutoff = pd.Timestamp("2025-01-10", tz="UTC")
    trades = pd.DataFrame({
        "entry_time": ["2025-01-08", "2025-01-09", "2025-01-11", "2025-01-12"],
        "exit_time": ["2025-01-09", "2025-01-11", "2025-01-12", "2025-01-13"],
    })
    train, hold = purge_crossing_trades(trades, cutoff, timedelta(days=2))
    assert list(train.index) == [0]
    assert list(hold.index) == [3]


def test_block_bootstrap_lower_bounds_for_positive_iid_sample():
    values = np.linspace(0.2, 1.0, 100)
    ci = block_bootstrap_ci(values, 5, iterations=500, seed=42)
    assert ci["mean_low"] > 0
    assert ci["trimmed_low"] > 0


def test_trial_registry_is_append_only_and_counted(tmp_path):
    path = tmp_path / "trials.jsonl"
    for i in range(10):
        append_trial({"config_hash": str(i), "dataset_id": "d", "metrics": {"x": i}}, path)
    assert count_trials(path) == 10
    assert len(path.read_text().splitlines()) == 10
    assert deflated_sharpe_probability(1.5, trials=10, observations=100) > \
           deflated_sharpe_probability(1.5, trials=100, observations=100)
    assert all(json.loads(line)["dataset_id"] == "d" for line in path.read_text().splitlines())


def test_gate_rejects_fewer_than_one_hundred_trades(tmp_path):
    cfg = {
        "strategy": {"alpha_gate": {"enabled": True, "record_path": str(tmp_path / "r.json")}},
    }
    (tmp_path / "r.json").write_text(json.dumps({
        "fingerprint": alpha_gate.config_fingerprint(cfg),
        "passed": True, "engine": "agent_engine",
        "provenance": {"dataset_id": None}, "metrics": {"trades": 40},
    }))
    assert alpha_gate.check_entries(cfg).reason == "insufficient_trades"


def test_fingerprint_changes_when_implementation_changes(tmp_path, monkeypatch):
    import app.risk.alpha_gate as gate

    cfg = {"strategy": {}}
    before = gate.config_fingerprint(cfg)
    monkeypatch.setattr(gate, "implementation_hash", lambda: "different-code")
    assert gate.config_fingerprint(cfg) != before


def test_gate_trade_count_aggregates_partial_fills_and_purges_crossing():
    from scripts.alpha_gate_run import _trade_returns

    details = [
        # Training position establishes a one-hour embargo.
        {"symbol": "A", "side": "buy", "qty": 1, "notional": 100,
         "commission": .05, "entry_time": "2025-01-08T10:00:00+00:00",
         "timestamp": "2025-01-08T10:00:00+00:00"},
        {"symbol": "A", "side": "sell", "qty": 1, "notional": 101,
         "commission": .05, "entry_time": "2025-01-08T10:00:00+00:00",
         "timestamp": "2025-01-08T11:00:00+00:00"},
        # This single position closes in two separate sell fills but is one trade.
        {"symbol": "A", "side": "buy", "qty": 10, "notional": 1000,
         "commission": .5, "entry_time": "2025-01-11T10:00:00+00:00",
         "timestamp": "2025-01-11T10:00:00+00:00"},
        {"symbol": "A", "side": "sell", "qty": 4, "notional": 440,
         "commission": .22, "entry_time": "2025-01-11T10:00:00+00:00",
         "timestamp": "2025-01-11T10:30:00+00:00"},
        {"symbol": "A", "side": "sell", "qty": 6, "notional": 630,
         "commission": .315, "entry_time": "2025-01-11T10:00:00+00:00",
         "timestamp": "2025-01-11T11:00:00+00:00"},
        # Crosses the calendar cutoff and must not enter holdout metrics.
        {"symbol": "B", "side": "buy", "qty": 1, "notional": 100,
         "commission": .05, "entry_time": "2025-01-09T10:00:00+00:00",
         "timestamp": "2025-01-09T10:00:00+00:00"},
        {"symbol": "B", "side": "sell", "qty": 1, "notional": 120,
         "commission": .06, "entry_time": "2025-01-09T10:00:00+00:00",
         "timestamp": "2025-01-10T10:00:00+00:00"},
    ]
    returns, durations, embargo, _ = _trade_returns(
        details, pd.Timestamp("2025-01-10T00:00:00Z")
    )
    assert len(returns) == 1
    assert returns[0] == pytest.approx((1070 - .535 - 1000.5) / 1000.5 * 100)
    assert durations == [60]
    assert embargo == timedelta(minutes=60)


def test_sim_broker_rejects_overspend_and_short_and_keeps_limit_open():
    pytest.importorskip("numpy")
    from app.backtest.agent_engine import FeeModel, FillModel, SimBroker

    broker = SimBroker(100, commission_pct=0.05,
                       fee_model=FeeModel(taker_pct=0.05, maker_pct=0.01),
                       fill_model=FillModel())
    broker.current_prices["BTC/USD"] = 10
    with pytest.raises(ValueError, match="insufficient cash"):
        broker.place_order("BTC/USD", "buy", 11, "market")
    with pytest.raises(ValueError, match="uncovered short"):
        broker.place_order("BTC/USD", "sell", 1, "market")
    oid = broker.place_order("BTC/USD", "buy", 1, "limit", limit_price=9)
    assert broker.get_open_orders()[0]["order_id"] == oid
    assert broker.trades == 0
    broker.current_prices["BTC/USD"] = 8
    broker.process_pending_orders()
    assert broker.get_open_orders() == []
    assert broker.positions["BTC/USD"] == pytest.approx(1)
    assert broker.trade_log[0]["maker"] is True
    assert broker.trade_log[0]["fee_rate_pct"] == pytest.approx(0.01)


def test_sim_broker_market_and_maker_fees_are_distinct():
    from app.backtest.agent_engine import FeeModel, SimBroker

    broker = SimBroker(1000, 0.05, fee_model=FeeModel(taker_pct=0.05, maker_pct=0.01))
    broker.current_prices["X"] = 10
    broker.place_order("X", "buy", 1, "market")
    market_fee = broker.trade_log[-1]["commission"]
    oid = broker.place_order("X", "buy", 1, "limit", limit_price=9)
    assert broker.get_open_orders()[0]["order_id"] == oid
    broker.current_prices["X"] = 8
    broker.process_pending_orders()
    maker_fee = broker.trade_log[-1]["commission"]
    assert maker_fee < market_fee
    assert broker.trade_log[-1]["maker"] is True


def test_fast_screen_uses_next_open_and_high_for_peak_mark():
    from scripts.alpha_validation import Costs, ExitLadder, simulate

    arr = {
        "close": np.array([100.0, 111.0, 112.0]),
        "open": np.array([99.0, 110.0, 112.0]),
        "high": np.array([101.0, 115.0, 113.0]),
        "low": np.array([98.0, 109.0, 111.0]),
        "sma": np.array([0.0, 0.0, 0.0]),
        "upper": np.array([0.0, 0.0, 0.0]),
        "rsi": np.array([50.0, 50.0, 50.0]),
        "entries": np.array([0]),
    }
    ladder = ExitLadder(
        hard_stop=0, trailing=0, take_profit=0, partial_tp=0,
        sma_exit_min_profit=-1, min_hold_minutes=0, min_hold_bars=1,
        use_peak_detect=False, max_hold_bars=1,
    )
    trade = simulate(arr, ladder, Costs(0, 0, 0))[0]
    # Entry is the next bar's open (110), not the signal bar close (100).
    assert trade["gross_pct"] == pytest.approx((112 / 110 - 1) * 100)
    assert trade["bars"] == 1

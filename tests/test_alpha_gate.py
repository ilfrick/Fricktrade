# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Alpha gate: flat unless proven otherwise.

The gate blocks NEW entries when the running strategy configuration has no
recorded hold-out pass. Exits must never be blocked — an unvalidated system
still has to be able to close what it already holds.
"""

import json

import pytest

from app.risk import alpha_gate


def _cfg(tmp_path, enabled=True, **overrides):
    cfg = {
        "strategy": {
            "names": ["crypto_mean_reversion"],
            "combine": "weighted",
            "weights": {"crypto_mean_reversion": 1.0},
            "min_conviction": 0.25,
            "params": {"crypto_mean_reversion": {"rsi_period": 70, "rsi_oversold": 30.0}},
            "alpha_gate": {
                "enabled": enabled,
                "record_path": str(tmp_path / "alpha_validation.json"),
            },
        },
        "data": {"interval": "1m"},
        "risk": {"crypto": {"hard_stop_pct": 5.0}},
    }
    cfg.update(overrides)
    return cfg


def _write(cfg, passed=True, fingerprint=None):
    path = alpha_gate.record_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fingerprint": fingerprint or alpha_gate.config_fingerprint(cfg),
        "passed": passed,
        "metrics": {},
    }))


class TestFingerprint:
    def test_stable_across_calls(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert alpha_gate.config_fingerprint(cfg) == alpha_gate.config_fingerprint(cfg)

    def test_changes_when_strategy_params_change(self, tmp_path):
        a = _cfg(tmp_path)
        b = _cfg(tmp_path)
        b["strategy"]["params"]["crypto_mean_reversion"]["rsi_oversold"] = 25.0
        assert alpha_gate.config_fingerprint(a) != alpha_gate.config_fingerprint(b)

    def test_changes_when_stop_changes(self, tmp_path):
        a = _cfg(tmp_path)
        b = _cfg(tmp_path)
        b["risk"]["crypto"]["hard_stop_pct"] = 2.0
        assert alpha_gate.config_fingerprint(a) != alpha_gate.config_fingerprint(b)

    def test_ignores_unrelated_keys(self, tmp_path):
        a = _cfg(tmp_path)
        b = _cfg(tmp_path)
        b["logging"] = {"file_path": "/tmp/other.log"}
        b["app"] = {"log_level": "DEBUG"}
        assert alpha_gate.config_fingerprint(a) == alpha_gate.config_fingerprint(b)


class TestGate:
    def test_blocks_when_no_record(self, tmp_path):
        d = alpha_gate.check_entries(_cfg(tmp_path))
        assert d.blocked and d.reason == "no_validation_record"

    def test_blocks_when_validation_failed(self, tmp_path):
        cfg = _cfg(tmp_path)
        _write(cfg, passed=False)
        d = alpha_gate.check_entries(cfg)
        assert d.blocked and d.reason == "validation_failed"

    def test_blocks_when_config_changed(self, tmp_path):
        cfg = _cfg(tmp_path)
        _write(cfg, passed=True, fingerprint="stale0000")
        d = alpha_gate.check_entries(cfg)
        assert d.blocked and d.reason == "config_changed_since_validation"

    def test_allows_when_validated(self, tmp_path):
        cfg = _cfg(tmp_path)
        _write(cfg, passed=True)
        d = alpha_gate.check_entries(cfg)
        assert not d.blocked and d.reason == "validated"

    def test_disabled_gate_allows(self, tmp_path):
        d = alpha_gate.check_entries(_cfg(tmp_path, enabled=False))
        assert not d.blocked and d.reason == "gate_disabled"

    def test_fails_closed_on_corrupt_record(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = alpha_gate.record_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json")
        assert alpha_gate.check_entries(cfg).blocked

    def test_write_record_roundtrip(self, tmp_path):
        cfg = _cfg(tmp_path)
        alpha_gate.write_record(cfg, passed=True, metrics={"net_E": 0.5})
        d = alpha_gate.check_entries(cfg)
        assert not d.blocked
        rec = alpha_gate.load_record(cfg)
        assert rec["metrics"]["net_E"] == 0.5
        assert rec["passed"] is True


class TestTraderIntegration:
    """The gate must stop entries but leave the exit path untouched."""

    def _agent(self, tmp_path, enabled=True):
        from unittest.mock import MagicMock, patch
        from app.agents.trader import TradingAgent
        broker = MagicMock()
        broker.get_account.return_value = {"equity": 10000, "cash": 10000}
        broker.get_positions.return_value = []
        broker.brokers = {"alpaca": broker}
        cfg = _cfg(tmp_path, enabled=enabled)
        cfg["risk"]["enabled"] = False
        cfg["learning"] = {}
        cfg["orchestrator"] = {"mode": "select", "rl": {"enabled": False}}
        cfg["brokers"] = {}
        cfg["execution"] = {}
        cfg["market"] = {}
        with patch("app.agents.trader.build_market_cache_config"), \
             patch("app.agents.trader.build_market_cache"):
            return TradingAgent(broker, cfg)

    def test_gate_blocks_buy_but_not_sell(self, tmp_path):
        agent = self._agent(tmp_path)
        # unvalidated config -> entries blocked
        assert alpha_gate.check_entries(agent.cfg).blocked
        # the exit path never consults the gate
        import inspect
        src = inspect.getsource(agent.run_once)
        assert 'action == "buy" and not _is_closing_position' in src, (
            "gate must be scoped to opening trades only"
        )

    def test_log_throttled(self, tmp_path):
        agent = self._agent(tmp_path)
        agent._log_alpha_gate_block("no_validation_record")
        first = dict(agent._alpha_gate_logged_at)
        agent._log_alpha_gate_block("no_validation_record")
        assert agent._alpha_gate_logged_at == first

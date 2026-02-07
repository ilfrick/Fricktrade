# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from app.agents.account_metrics import AccountMetricsUpdater, _var_cvar_from_history


class TestVarCvar:
    def test_insufficient_history(self):
        result = _var_cvar_from_history([100.0], 0.95)
        assert result["var_pct"] == 0.0
        assert result["cvar_pct"] == 0.0

    def test_basic_history(self):
        history = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]
        result = _var_cvar_from_history(history, 0.95)
        assert result["var_pct"] >= 0.0
        assert result["cvar_pct"] >= 0.0


class TestAccountMetricsUpdater:
    def test_init(self):
        updater = AccountMetricsUpdater({"risk": {"var": {"enabled": False}}})
        assert updater.equity_start is None
        assert updater.equity_peak is None
        assert updater.current_drawdown_pct == 0.0

    def test_equity_setters(self):
        updater = AccountMetricsUpdater({})
        updater.equity_start = 100000.0
        updater.equity_peak = 105000.0
        assert updater.equity_start == 100000.0
        assert updater.equity_peak == 105000.0

    def test_var_limit_no_var_config(self):
        updater = AccountMetricsUpdater({"risk": {}})

        class FakeBrokerState:
            var_cvar = {}

        result = updater.var_limit_reason(FakeBrokerState())
        assert result is None

    def test_var_limit_enabled_but_within(self):
        updater = AccountMetricsUpdater({
            "risk": {"var": {"enabled": True, "max_var_pct": 5.0, "max_cvar_pct": 10.0}}
        })

        class FakeBrokerState:
            var_cvar = {"var_pct": 1.0, "cvar_pct": 2.0}

        result = updater.var_limit_reason(FakeBrokerState())
        assert result is None

    def test_var_limit_exceeded(self):
        updater = AccountMetricsUpdater({
            "risk": {"var": {"enabled": True, "max_var_pct": 2.0, "max_cvar_pct": 5.0}}
        })
        # Simulate exceeding global var
        updater._var_cvar = {"var_pct": 3.0, "cvar_pct": 1.0}

        class FakeBrokerState:
            var_cvar = {"var_pct": 1.0, "cvar_pct": 1.0}

        result = updater.var_limit_reason(FakeBrokerState())
        assert result == "var_limit"

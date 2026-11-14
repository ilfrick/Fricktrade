# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Phase 0 of docs/PLAN-2026-Q4.md — security and correctness hardening.

Each test maps to one acceptance criterion of a Phase 0 task:

0.1 API auth fails closed, constant-time compare
0.2 every credential is masked, and masking survives a UI round-trip
0.3 config/checkpoint writes are atomic
0.4 the daily-loss budget survives a restart, on one clock
0.5 config keys have consumers
0.6 the alpha gate cannot be disabled by deleting config
0.7 a failed holiday scrape does not wipe the calendar
0.8 healthwatch scheduler cannot die silently; stop_services semantics
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.risk import alpha_gate
from app.risk.manager import RiskManager
from app.utils.atomic import atomic_write_text


# ---------------------------------------------------------------------------
# 0.1 — API authentication fails closed
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.api import server  # noqa: E402


def _write_api_cfg(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


class TestApiAuthFailsClosed:
    def test_missing_config_is_503_not_open(self, tmp_path, monkeypatch):
        """A missing config volume must deny, not silently disable auth.

        The old _load_raw_config returned {} on FileNotFoundError, which made
        _resolve_auth_token return None and _require_auth return immediately.
        """
        monkeypatch.setattr(server, "CONFIG_PATH", tmp_path / "does-not-exist.yaml")
        client = TestClient(server.app)
        assert client.get("/config/raw").status_code == 503

    def test_corrupt_config_is_503(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("api: [unclosed\n", encoding="utf-8")
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        client = TestClient(server.app)
        assert client.get("/config/raw").status_code == 503

    def test_absent_auth_block_requires_token(self, tmp_path, monkeypatch):
        """Deleting the api.auth block must NOT open the API."""
        cfg_path = tmp_path / "config.yaml"
        _write_api_cfg(cfg_path, {"market": {"venue": "Multi"}})
        monkeypatch.setenv("FRICKTRADE_API_TOKEN", "tok")
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        client = TestClient(server.app)
        assert client.get("/config/raw").status_code == 401
        assert client.get("/config/raw", headers={"X-API-Key": "tok"}).status_code == 200

    def test_enabled_without_token_refuses(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        _write_api_cfg(cfg_path, {"api": {"auth": {"enabled": True, "token_env": "MISSING_TOK"}}})
        monkeypatch.delenv("MISSING_TOK", raising=False)
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        client = TestClient(server.app)
        assert client.get("/config/raw").status_code == 500

    def test_explicit_opt_out_still_works(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        _write_api_cfg(cfg_path, {"api": {"auth": {"enabled": False}}})
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        client = TestClient(server.app)
        assert client.get("/config/raw").status_code == 200

    def test_restart_requires_auth(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        _write_api_cfg(cfg_path, {"api": {"auth": {"enabled": True, "token_env": "TOK2"}}})
        monkeypatch.setenv("TOK2", "s3cret")
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        monkeypatch.setattr(server, "restart_flag_path", lambda: tmp_path / "restart.flag")
        client = TestClient(server.app)
        assert client.post("/restart").status_code == 401
        assert client.post("/restart", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/restart", headers={"Authorization": "Bearer s3cret"}).status_code == 200


# ---------------------------------------------------------------------------
# 0.2 — credential masking
# ---------------------------------------------------------------------------

_LEAKY_CFG = {
    "api": {"auth": {"enabled": True, "token_env": "FRICKTRADE_API_TOKEN"}},
    "brokers": {
        "binance": {"api_key": "BN_KEY", "api_secret": "BN_SECRET", "accounts": []},
        "alpaca": {"api_key": "AL_KEY", "api_secret": "AL_SECRET", "accounts": []},
        "ibkr": {"password": "IB_PW", "accounts": []},
    },
    "fred": {"api_key": "FRED_KEY", "base_url": "https://api.stlouisfed.org/fred"},
    "alt_data": {
        "coinglass": {"api_key": "CG_KEY"},
        "alpha_vantage": {"api_key": "AV_KEY"},
    },
    "kill_switch": {"armed": False, "required_code": "INTERLOCK"},
    "monitoring": {"audit": {"signing": {"secret_env": "AUDIT_SIGNING_SECRET"}}},
    "news": {"sources": [{"name": "cryptopanic", "api_key": "CP_KEY"}]},
}

_SECRET_VALUES = [
    "BN_KEY", "BN_SECRET", "AL_KEY", "AL_SECRET", "IB_PW",
    "FRED_KEY", "CG_KEY", "AV_KEY", "INTERLOCK", "CP_KEY",
]


class TestSecretMasking:
    def test_every_credential_is_masked(self):
        cfg = json.loads(json.dumps(_LEAKY_CFG))
        server._mask_secrets(cfg)
        serialised = json.dumps(cfg)
        for secret in _SECRET_VALUES:
            assert secret not in serialised, f"{secret} leaked through /config"

    def test_env_var_names_are_not_masked(self):
        """`token_env` / `secret_env` name a variable; they are not credentials."""
        cfg = json.loads(json.dumps(_LEAKY_CFG))
        server._mask_secrets(cfg)
        assert cfg["api"]["auth"]["token_env"] == "FRICKTRADE_API_TOKEN"
        assert cfg["monitoring"]["audit"]["signing"]["secret_env"] == "AUDIT_SIGNING_SECRET"

    def test_non_secret_preserved(self):
        cfg = json.loads(json.dumps(_LEAKY_CFG))
        server._mask_secrets(cfg)
        assert cfg["fred"]["base_url"] == "https://api.stlouisfed.org/fred"
        assert cfg["kill_switch"]["armed"] is False

    def test_round_trip_does_not_write_mask_to_disk(self):
        """A UI round-trip (GET masked -> POST back) must restore real values.

        Otherwise masking turns into credential destruction on the next save.
        """
        original = json.loads(json.dumps(_LEAKY_CFG))
        posted = json.loads(json.dumps(_LEAKY_CFG))
        server._mask_secrets(posted)
        server._merge_secrets(posted, original)
        assert posted == original

    def test_cleared_key_falls_back_to_existing(self):
        original = {"brokers": {"binance": {"api_key": "REAL"}}}
        posted = {"brokers": {"binance": {"api_key": None}}}
        server._merge_secrets(posted, original)
        assert posted["brokers"]["binance"]["api_key"] == "REAL"


# ---------------------------------------------------------------------------
# 0.3 — atomic writes
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_writes_content(self, tmp_path):
        target = tmp_path / "a.txt"
        atomic_write_text(target, "hello")
        assert target.read_text() == "hello"

    def test_no_temp_files_left(self, tmp_path):
        target = tmp_path / "b.txt"
        atomic_write_text(target, "x")
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "b.txt"]
        assert leftovers == []

    def test_failure_preserves_previous_content(self, tmp_path):
        """A crash mid-write must leave the old file readable, not truncated."""
        target = tmp_path / "config.yaml"
        target.write_text("original: true\n")
        with patch("app.utils.atomic.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                atomic_write_text(target, "partially-writ")
        assert target.read_text() == "original: true\n"
        assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]

    def test_checkpoint_uses_atomic_write(self, tmp_path):
        from app.utils.checkpoint import load_checkpoint, maybe_save_checkpoint

        cfg = {
            "checkpointing": {
                "enabled": True,
                "interval_seconds": 0,
                "dir": str(tmp_path),
                "keep_history": False,
                "retention": {"max_files": 5, "max_age_hours": 0},
            }
        }
        maybe_save_checkpoint("trader", {"k": 1}, cfg, None)
        assert load_checkpoint("trader", cfg)["payload"] == {"k": 1}
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 0.4 — the daily-loss budget survives a restart, on one clock
# ---------------------------------------------------------------------------

class TestDailyLossPersistence:
    def test_risk_manager_exposes_its_own_day(self):
        from zoneinfo import ZoneInfo

        rm = RiskManager({"enabled": True, "max_daily_loss_pct": 3.0}, tz=ZoneInfo("US/Eastern"))
        assert rm.today() == datetime.now(rm._tz).date()

    def test_account_metrics_uses_the_risk_timezone(self):
        """A day baseline on a different clock than the accumulator cleared the cap
        four hours early (midnight UTC vs midnight US/Eastern)."""
        from app.agents.account_metrics import AccountMetricsUpdater

        sentinel = date(1999, 12, 31)
        fake_risk = MagicMock()
        fake_risk.today.return_value = sentinel
        updater = AccountMetricsUpdater({})
        assert updater._today({"alpaca": MagicMock(risk=fake_risk)}) == sentinel
        # no broker states -> UTC fallback, not an exception
        assert updater._today({}) == datetime.now(timezone.utc).date()

    def test_daily_loss_survives_checkpoint_round_trip(self, tmp_path):
        agent = _make_agent(log_dir=tmp_path, day_start_equity=1000.0, day_pnl=-4.0)
        agent._maybe_checkpoint()

        # The payload itself must carry the baseline, otherwise the restore cannot.
        saved = json.loads((tmp_path / "trader.json").read_text())["payload"]
        stored = saved["broker_states"]["alpaca"]
        assert stored["day_start_equity"] == pytest.approx(1000.0)
        assert stored["day_pnl_pct"] == pytest.approx(-4.0)
        assert stored["day_start_date"] is not None

        # TradingAgent.__init__ restores the checkpoint.
        fresh = _make_agent(log_dir=tmp_path)
        restored = fresh._broker_state("alpaca")
        assert restored.day_start_equity == pytest.approx(1000.0)
        assert isinstance(restored.day_start_date, date)
        assert restored.day_pnl_pct == pytest.approx(-4.0)
        # The accumulator must be re-armed, not zeroed: a restart during a -4% day
        # used to hand back the whole daily budget.
        assert restored.risk.daily_loss == pytest.approx(-4.0)

    def test_day_baseline_resets_only_on_new_day(self, tmp_path):
        agent = _make_agent(log_dir=tmp_path)
        state = agent._broker_state("alpaca")
        state.day_start_date = date(2020, 1, 1)
        state.day_start_equity = 500.0
        agent._account_metrics.update_broker_equity("alpaca", state, 480.0, 0.0)
        # A stale baseline re-baselines to current equity (new day), and the day P&L
        # reported to the risk manager is then 0 rather than -4%.
        assert state.day_start_equity == pytest.approx(480.0)
        assert state.day_start_date != date(2020, 1, 1)
        assert state.day_pnl_pct == pytest.approx(0.0)

    def test_same_day_keeps_the_baseline(self, tmp_path):
        agent = _make_agent(log_dir=tmp_path)
        state = agent._broker_state("alpaca")
        state.day_start_date = state.risk.today()
        state.day_start_equity = 500.0
        agent._account_metrics.update_broker_equity("alpaca", state, 480.0, 0.0)
        assert state.day_start_equity == pytest.approx(500.0)
        assert state.day_pnl_pct == pytest.approx(-4.0)
        assert state.risk.daily_loss == pytest.approx(-4.0)


def _make_agent(log_dir=None, day_start_equity=None, day_pnl=None):
    from app.agents.trader import TradingAgent

    broker = MagicMock()
    broker.get_account.return_value = {"equity": 10000, "cash": 10000}
    broker.get_positions.return_value = []
    broker.brokers = {"alpaca": broker}
    cfg = {
        "strategy": {
            "names": ["crypto_mean_reversion"],
            "combine": "weighted",
            "weights": {"crypto_mean_reversion": 1.0},
            "min_conviction": 0.25,
            "params": {"crypto_mean_reversion": {}},
            "alpha_gate": {"enabled": True, "record_path": str(Path(log_dir or ".") / "av.json")},
        },
        "risk": {"enabled": True, "vol_targeting": {"enabled": False}},
        "data": {"interval": "1m"},
        "learning": {},
        "orchestrator": {"mode": "select", "rl": {"enabled": False}},
        "brokers": {},
        "execution": {},
        "market": {},
        "checkpointing": {
            "enabled": True,
            "interval_seconds": 0,
            "dir": str(log_dir),
            "keep_history": False,
            "retention": {"max_files": 5, "max_age_hours": 0},
        },
    }
    with patch("app.agents.trader.build_market_cache_config"), \
         patch("app.agents.trader.build_market_cache"):
        agent = TradingAgent(broker, cfg)
    if day_start_equity is not None:
        state = agent._broker_state("alpaca")
        state.day_start_equity = day_start_equity
        state.day_start_date = date.today()
        state.day_pnl_pct = day_pnl
        state.risk.update_daily_loss(day_pnl)
    return agent


# ---------------------------------------------------------------------------
# 0.5 — config keys have consumers
# ---------------------------------------------------------------------------

# Curated list of keys that materially change trading behaviour. Extend it whenever a
# key is found to be decorative — that is exactly the bug class this guards against
# (three instances found in the 2026-11-14 audit: hard_stop_pct, trailing_stop_pct,
# max_daily_loss_pct were all configured and never read).
_MUST_HAVE_CONSUMER = {
    "risk.max_daily_loss_pct": "max_daily_loss_pct",
    "risk.crypto.hard_stop_pct": "hard_stop_pct",
    "risk.crypto.trailing_stop_pct": "trailing_stop_pct",
    "risk.crypto.take_profit_pct": "take_profit_pct",
    "risk.crypto.max_position_size_pct": "max_position_size_pct",
    "strategy.min_conviction": "min_conviction",
    "trading_limits.min_notional": "min_notional",
}


class TestConfigKeysHaveConsumers:
    """Guard against decorative config: keys that look tunable and change nothing.

    Limitation, stated honestly: this is a static word-boundary scan. A key mentioned
    only in a comment passes. It catches the observed failure mode — a key that exists
    in config.yaml and appears nowhere in the code at all — which is how
    `hard_stop_pct`, `trailing_stop_pct` and `max_daily_loss_pct` stayed "live" for
    months. Behavioural tests are the real guard; this is a cheap tripwire.
    """

    @pytest.mark.parametrize("dotted,literal", sorted(_MUST_HAVE_CONSUMER.items()))
    def test_key_is_read_somewhere(self, dotted, literal):
        import re

        root = Path(__file__).resolve().parents[1] / "app"
        pattern = re.compile(rf"\b{re.escape(literal)}\b")
        hits = [
            str(path.relative_to(root))
            for path in root.rglob("*.py")
            if "__pycache__" not in str(path)
            and pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
        ]
        assert hits, f"{dotted} is configured but {literal!r} appears nowhere in app/"

    def test_min_notional_is_enforced_by_preflight(self):
        """check_order_limits used to read `min_order_notional`, a key that never
        existed, so the whole preflight was a no-op."""
        rm = RiskManager({"enabled": True})
        limits = {"enabled": True, "min_notional": 10.0}
        violates, reason = rm.check_order_limits(1, 5.0, limits)  # $5 notional
        assert violates and reason == "order_limit"
        violates, _ = rm.check_order_limits(10, 5.0, limits)  # $50 notional
        assert not violates

    def test_removed_crypto_daily_loss_key_is_not_resurrected(self):
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "config.yaml").read_text()
        )
        assert "max_daily_loss_pct" not in cfg["risk"]["crypto"], (
            "risk.crypto.max_daily_loss_pct is dead config: RiskManager only reads the "
            "global key. Re-adding it without wiring it up is silently ignored."
        )
        assert cfg["risk"]["max_daily_loss_pct"] > 0


# ---------------------------------------------------------------------------
# 0.6 — the alpha gate cannot be disabled by omission
# ---------------------------------------------------------------------------

class TestAlphaGateFailsClosed:
    def test_absent_block_blocks_entries(self, tmp_path):
        cfg = {"strategy": {"names": ["crypto_mean_reversion"]}}
        assert alpha_gate.check_entries(cfg).blocked
        assert alpha_gate.check_entries(cfg).reason == "no_validation_record"

    def test_explicit_disable_is_respected(self, tmp_path):
        cfg = {"strategy": {"alpha_gate": {"enabled": False}}}
        decision = alpha_gate.check_entries(cfg)
        assert not decision.blocked
        assert decision.reason == "gate_disabled"

    def test_config_update_cannot_remove_the_gate(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        full = {
            "api": {"auth": {"enabled": False}},
            "strategy": {"alpha_gate": {"enabled": True, "record_path": "x"}},
        }
        _write_api_cfg(cfg_path, full)
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        client = TestClient(server.app)
        stripped = yaml.safe_dump({"api": {"auth": {"enabled": False}}, "strategy": {}})
        resp = client.post("/config/update", json={"yaml": stripped})
        assert resp.status_code == 400
        assert "alpha_gate" in resp.json()["detail"]
        # and the file on disk is untouched
        assert "alpha_gate" in cfg_path.read_text()

    def test_config_update_rejects_type_clobber(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        _write_api_cfg(cfg_path, {"api": {"auth": {"enabled": False}}, "risk": {"enabled": True}})
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        client = TestClient(server.app)
        resp = client.post(
            "/config/update",
            json={"yaml": yaml.safe_dump({"api": {"auth": {"enabled": False}}, "risk": None})},
        )
        assert resp.status_code == 400

    def test_config_update_is_atomic(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        _write_api_cfg(cfg_path, {"api": {"auth": {"enabled": False}}, "app": {"env": "prod"}})
        monkeypatch.setattr(server, "CONFIG_PATH", cfg_path)
        client = TestClient(server.app)
        resp = client.post(
            "/config/update",
            json={"yaml": yaml.safe_dump({"api": {"auth": {"enabled": False}}, "app": {"env": "dev"}})},
        )
        assert resp.status_code == 200
        assert yaml.safe_load(cfg_path.read_text())["app"]["env"] == "dev"
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 0.7 — a failed holiday scrape must not wipe the calendar
# ---------------------------------------------------------------------------

class TestHolidayRefreshGuard:
    def test_empty_scrape_does_not_wipe_existing(self):
        from app.utils.holiday_update import _holidays_look_plausible

        existing = ["2026-01-01", "2026-04-03", "2026-05-25", "2026-07-03"]
        assert not _holidays_look_plausible([], existing, date(2026, 1, 1), date(2026, 12, 31))

    def test_empty_scrape_ok_when_nothing_to_lose(self):
        from app.utils.holiday_update import _holidays_look_plausible

        assert _holidays_look_plausible([], [], date(2026, 1, 1), date(2026, 12, 31))

    def test_truncated_parse_is_rejected(self):
        from app.utils.holiday_update import _holidays_look_plausible

        existing = [f"2026-0{m}-01" for m in range(1, 9)]
        assert not _holidays_look_plausible(
            ["2026-01-01"], existing, date(2026, 1, 1), date(2026, 12, 31)
        )

    def test_full_list_is_accepted(self):
        from app.utils.holiday_update import _holidays_look_plausible

        existing = ["2026-01-01", "2026-04-03"]
        assert _holidays_look_plausible(
            ["2026-01-01", "2026-04-03"], existing, date(2026, 1, 1), date(2026, 12, 31)
        )


# ---------------------------------------------------------------------------
# 0.8 — healthwatch scheduler
# ---------------------------------------------------------------------------

class TestHealthwatchStopServices:
    def test_absent_stop_list_stops_nothing(self):
        """`stop_services: null` used to mean `services - keep_services`, i.e. every
        service not explicitly whitelisted."""
        from app.monitoring.healthwatch import _resolve_stop_services

        keep = {"trader", "api"}
        assert _resolve_stop_services({"stop_services": None}, keep) == (set(), set())
        assert _resolve_stop_services({}, keep) == (set(), set())
        assert _resolve_stop_services({"stop_services": []}, keep) == (set(), set())

    def test_explicit_list_still_works(self):
        from app.monitoring.healthwatch import _resolve_stop_services

        stop, conflicts = _resolve_stop_services({"stop_services": ["ollama"]}, {"trader"})
        assert stop == {"ollama"}
        assert conflicts == set()

    def test_keep_services_wins_on_conflict(self):
        from app.monitoring.healthwatch import _resolve_stop_services

        stop, conflicts = _resolve_stop_services({"stop_services": ["ollama", "trader"]}, {"trader"})
        assert stop == {"ollama"}
        assert conflicts == {"trader"}

    def test_scheduler_wrapper_survives_exceptions(self):
        """The scheduler runs in a daemon thread; an escaping exception used to kill it
        while /metrics kept reporting healthwatch_up=1."""
        import app.monitoring.healthwatch as hw

        attempts = {"n": 0}

        def flaky(cfg):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("docker socket went away")
            return None

        with patch.object(hw, "_market_scheduler_loop", side_effect=flaky), \
             patch.object(hw.time, "sleep", lambda _s: None):
            hw._run_market_scheduler({})  # must not raise

        assert attempts["n"] == 2, "wrapper must retry after a crash"

    def test_tcp_server_allows_reuse(self):
        import app.monitoring.healthwatch as hw

        assert hw._ReusableTCPServer.allow_reuse_address is True

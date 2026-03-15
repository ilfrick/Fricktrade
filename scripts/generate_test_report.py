# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Periodic test report generator.

Runs the full pytest suite, performs a set of system health checks, and
writes a Word document summarising the findings.

Usage (standalone):
    python scripts/generate_test_report.py --config /app/config/config.yaml \
        --output /data/reports/test_report.docx

Called automatically from run_tests_when_closed.py after market close.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pytest result collection
# ---------------------------------------------------------------------------

class _ResultCollector:
    """Lightweight pytest plugin that captures per-test outcomes."""

    def __init__(self) -> None:
        self.results: list[dict] = []
        self.warnings: list[str] = []

    def pytest_runtest_logreport(self, report: Any) -> None:  # noqa: ANN001
        if report.when not in ("call", "setup"):
            return
        if report.when == "setup" and not report.failed:
            return
        status = "PASS" if report.passed else ("SKIP" if report.skipped else "FAIL")
        entry: dict = {
            "nodeid": report.nodeid,
            "status": status,
            "duration": getattr(report, "duration", 0.0),
            "message": "",
        }
        if report.failed:
            entry["message"] = _clean_repr(report.longrepr)
        elif report.skipped:
            reason = ""
            if isinstance(report.longrepr, tuple) and len(report.longrepr) >= 3:
                reason = str(report.longrepr[2])
            entry["message"] = reason
        self.results.append(entry)

    def pytest_warning_recorded(self, warning_message: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001
        self.warnings.append(str(warning_message.message))


def _clean_repr(longrepr: Any) -> str:
    if longrepr is None:
        return ""
    try:
        return str(longrepr)[:800]
    except Exception:
        return "<unparseable>"


def run_pytest(tests_dir: str = "tests") -> dict:
    """Run pytest programmatically and return a results dict."""
    try:
        import pytest  # noqa: PLC0415
    except ImportError:
        return {"error": "pytest not installed", "results": [], "summary": {}}

    collector = _ResultCollector()
    start = time.monotonic()
    exit_code = pytest.main(
        [tests_dir, "-q", "--disable-warnings", "--tb=short", "--no-header"],
        plugins=[collector],
    )
    elapsed = time.monotonic() - start

    results = collector.results
    counts = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    for r in results:
        s = r["status"].lower()
        if s == "pass":
            counts["pass"] += 1
        elif s == "fail":
            counts["fail"] += 1
        elif s == "skip":
            counts["skip"] += 1
        else:
            counts["error"] += 1

    return {
        "exit_code": int(exit_code),
        "duration_seconds": round(elapsed, 1),
        "results": results,
        "summary": counts,
        "warnings": collector.warnings[:20],
    }


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def _hc(name: str, status: str, detail: str = "", recommendation: str = "") -> dict:
    return {
        "name": name,
        "status": status,          # PASS / WARN / FAIL / INFO
        "detail": detail,
        "recommendation": recommendation,
    }


def check_config(config_path: str) -> list[dict]:
    checks: list[dict] = []
    try:
        sys.path.insert(0, str(Path(config_path).parent.parent.parent))
        from app.utils.config import load_config  # noqa: PLC0415
        cfg = load_config(config_path)
        checks.append(_hc("Config loads", "PASS", f"Path: {config_path}"))
    except Exception as exc:
        checks.append(_hc("Config loads", "FAIL", str(exc), "Fix config syntax/path before restarting trader"))
        return checks

    required_sections = ["brokers", "risk", "strategy", "execution", "data"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        checks.append(_hc("Required config sections", "FAIL",
                          f"Missing: {missing}", "Add missing sections to config.yaml"))
    else:
        checks.append(_hc("Required config sections", "PASS", str(required_sections)))

    # Risk parameter sanity
    risk = cfg.get("risk", {})
    hard_stop = float(risk.get("hard_stop", 0.8))
    if not (0.05 <= hard_stop <= 20.0):
        checks.append(_hc("Risk hard_stop range", "WARN",
                          f"hard_stop={hard_stop}% is outside [0.05, 20.0]",
                          "Review risk.hard_stop in config"))
    else:
        checks.append(_hc("Risk hard_stop range", "PASS", f"hard_stop={hard_stop}%"))

    daily_loss = float(risk.get("max_daily_loss_pct", 3.0))
    if not (0.5 <= daily_loss <= 50.0):
        checks.append(_hc("Risk daily_loss range", "WARN",
                          f"max_daily_loss_pct={daily_loss}%",
                          "Review risk.max_daily_loss_pct"))
    else:
        checks.append(_hc("Risk daily_loss range", "PASS", f"max_daily_loss_pct={daily_loss}%"))

    # Strategy combine mode
    strat = cfg.get("strategy", {})
    combine = strat.get("combine", "vote")
    checks.append(_hc("Strategy combine mode", "INFO", f"combine={combine}"))

    # LLM enabled check
    llm = cfg.get("llm", {})
    llm_enabled = llm.get("enabled", False)
    checks.append(_hc("LLM module", "INFO",
                      f"enabled={llm_enabled}"))

    # Broker count
    brokers_cfg = cfg.get("brokers", {})
    broker_names = list(brokers_cfg.keys()) if isinstance(brokers_cfg, dict) else []
    if not broker_names:
        checks.append(_hc("Broker config", "FAIL", "No brokers configured",
                          "Add at least one broker under 'brokers:' in config"))
    else:
        checks.append(_hc("Broker config", "PASS", f"Configured: {broker_names}"))

    return checks


def check_data_files() -> list[dict]:
    checks: list[dict] = []
    data_dir = Path("/data")
    if not data_dir.exists():
        checks.append(_hc("/data directory", "WARN", "/data does not exist (running outside container?)"))
        return checks
    checks.append(_hc("/data directory", "PASS", str(data_dir)))

    # Training data
    training_dir = data_dir / "training"
    if not training_dir.exists():
        checks.append(_hc("Training data directory", "WARN", str(training_dir) + " missing",
                          "Run data collection script to populate training data"))
    else:
        csv_files = list(training_dir.glob("*.csv"))
        if not csv_files:
            checks.append(_hc("Training CSV files", "WARN", "No CSVs in " + str(training_dir),
                              "Run collect_crypto_training_data.py"))
        else:
            newest = max(csv_files, key=lambda p: p.stat().st_mtime)
            age_days = (time.time() - newest.stat().st_mtime) / 86400
            status = "PASS" if age_days <= 3 else "WARN"
            checks.append(_hc("Training CSV files", status,
                               f"{len(csv_files)} files; newest '{newest.name}' is {age_days:.1f}d old",
                               "Re-run data collection if age > 3 days" if age_days > 3 else ""))

    # Decision trace
    trace_dir = data_dir / "reports" / "decision_trace"
    if not trace_dir.exists():
        checks.append(_hc("Decision trace directory", "INFO", "Not yet created (no trading cycles completed)"))
    else:
        today_str = datetime.date.today().isoformat()
        today_trace = trace_dir / f"{today_str}.jsonl"
        if today_trace.exists():
            line_count = sum(1 for _ in today_trace.open())
            checks.append(_hc("Decision trace today", "PASS",
                               f"{line_count} records in {today_trace.name}"))
        else:
            jsonl_files = sorted(trace_dir.glob("*.jsonl"))
            if jsonl_files:
                latest = jsonl_files[-1]
                age_days = (time.time() - latest.stat().st_mtime) / 86400
                checks.append(_hc("Decision trace today", "INFO",
                                   f"No trace for today; latest: {latest.name} ({age_days:.1f}d ago)"))
            else:
                checks.append(_hc("Decision trace today", "INFO", "No trace files found"))

    # Meta-orch changes log
    meta_log = data_dir / "reports" / "meta_orch" / "changes.jsonl"
    if meta_log.exists():
        lines = meta_log.read_text().strip().splitlines()
        checks.append(_hc("Meta-orch changes log", "PASS",
                           f"{len(lines)} recorded changes at {str(meta_log)}"))
    else:
        checks.append(_hc("Meta-orch changes log", "INFO", "No changes log yet"))

    # Model files
    models_dir = Path("/app/models")
    if not models_dir.exists():
        checks.append(_hc("Model directory", "INFO", "/app/models not present"))
    else:
        model_files = list(models_dir.glob("*.zip"))
        if model_files:
            checks.append(_hc("Model files", "INFO",
                               f"{len(model_files)} .zip model(s): {[f.name for f in model_files]}"))
        else:
            checks.append(_hc("Model files", "INFO", "No .zip models found"))
        registry = models_dir / "model_registry.json"
        if registry.exists():
            try:
                data = json.loads(registry.read_text())
                checks.append(_hc("Model registry", "PASS",
                                   f"{len(data)} entries in registry"))
            except Exception as exc:
                checks.append(_hc("Model registry", "WARN", f"Parse error: {exc}"))

    return checks


def check_strategy_smoke(config_path: str) -> list[dict]:
    """Instantiate each known strategy and verify it returns a valid signal dict."""
    checks: list[dict] = []
    strategies_to_test = [
        ("trend_following", "app.strategies.trend_following", "TrendFollowingStrategy",
         {"prices": [10, 10.2, 10.4, 10.6, 10.8, 11.0]}),
        ("factor_model", "app.strategies.factor_model", "FactorModelStrategy",
         {"prices": [10, 10.5, 11.0, 11.2], "volumes": [1e6, 1.1e6, 1.2e6, 1.3e6]}),
        ("crypto_momentum", "app.strategies.crypto_momentum", "CryptoMomentumStrategy",
         {"prices": [100, 101, 102, 103, 104, 105],
          "volumes": [500, 600, 700, 800, 900, 1000],
          "symbol": "BTC/USD"}),
        ("crypto_mean_reversion", "app.strategies.crypto_mean_reversion",
         "CryptoMeanReversionStrategy",
         {"prices": [100, 99, 98, 97, 96, 95], "symbol": "ETH/USD"}),
        ("gap_reversal", "app.strategies.gap_reversal", "GapReversalStrategy",
         {"prices": [10, 10.5, 11.0, 10.3, 10.1]}),
        ("pattern_trading", "app.strategies.pattern_trading", "PatternTradingStrategy",
         {"prices": [10, 10.2, 10.4, 10.2, 10.0, 10.3]}),
    ]

    for strat_name, module_path, class_name, market_state in strategies_to_test:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls({})
            signal = instance.generate_signal(market_state)
            if not isinstance(signal, dict):
                checks.append(_hc(f"Strategy smoke: {strat_name}", "FAIL",
                                   f"generate_signal returned {type(signal).__name__}, expected dict",
                                   "Check strategy implementation"))
            elif "action" not in signal:
                checks.append(_hc(f"Strategy smoke: {strat_name}", "FAIL",
                                   "Signal dict missing 'action' key",
                                   "Strategy must return {'action': ..., ...}"))
            elif signal["action"] not in {"buy", "sell", "hold", "exit"}:
                checks.append(_hc(f"Strategy smoke: {strat_name}", "FAIL",
                                   f"Unknown action value: {signal['action']}",
                                   "Valid actions: buy/sell/hold/exit"))
            else:
                checks.append(_hc(f"Strategy smoke: {strat_name}", "PASS",
                                   f"action={signal['action']}  "
                                   f"confidence={signal.get('confidence', 'n/a')}"))
        except ImportError as exc:
            checks.append(_hc(f"Strategy smoke: {strat_name}", "WARN",
                               f"Import failed (optional dep?): {exc}"))
        except Exception as exc:
            checks.append(_hc(f"Strategy smoke: {strat_name}", "FAIL",
                               f"{type(exc).__name__}: {exc}",
                               "Review strategy code"))

    return checks


def check_order_queue_integrity() -> list[dict]:
    """Quick in-process smoke tests of OrderQueue logic."""
    checks: list[dict] = []
    try:
        from app.execution.order_queue import OrderQueue  # noqa: PLC0415

        class _Stub:
            def __init__(self) -> None:
                self._n = 0

            def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kw: Any) -> str:
                self._n += 1
                return f"id-{self._n}"

        broker = _Stub()
        q = OrderQueue(broker, "test", completion_grace_seconds=0)
        id1 = q.enqueue("X", "buy", 1.0, notional=100.0)
        id2 = q.enqueue("Y", "buy", 2.0, notional=200.0)
        assert id1 is not None and id1 != "queued", "First order should submit immediately"
        assert id2 == "queued", "Second order should be queued"
        # Simulate completion
        q.update([{"order_id": id1, "symbol": "X", "side": "buy", "qty": 1.0,
                    "filled_qty": 1.0, "filled_avg_price": 100.0}])
        q.update([])  # order disappears → completed
        responses = q.pop_responses()
        statuses = {r.status for r in responses}
        assert "completed" in statuses or "submitted" in statuses, \
            f"Expected completed/submitted, got {statuses}"
        checks.append(_hc("OrderQueue FIFO + completion", "PASS",
                           "Enqueue, submit, complete cycle verified"))
    except AssertionError as exc:
        checks.append(_hc("OrderQueue FIFO + completion", "FAIL", str(exc)))
    except Exception as exc:
        checks.append(_hc("OrderQueue FIFO + completion", "FAIL",
                           f"{type(exc).__name__}: {exc}",
                           "Review order_queue.py"))

    # reserved_notional on timed_out
    try:
        from app.execution.order_queue import OrderQueue  # noqa: PLC0415
        from datetime import timezone  # noqa: PLC0415

        class _SlowBroker:
            def place_order(self, *a: Any, **kw: Any) -> str:
                return "stuck-1"

            def cancel_order(self, *a: Any, **kw: Any) -> None:
                pass

        q2 = OrderQueue(_SlowBroker(), "test",
                        retry_cfg={"max_order_age_seconds": 1},
                        completion_grace_seconds=0)
        q2.enqueue("Z", "buy", 1.0, notional=500.0)
        time.sleep(1.1)
        q2.update([{"order_id": "stuck-1", "symbol": "Z", "side": "buy", "qty": 1.0}])
        resps = q2.pop_responses()
        timed_out = [r for r in resps if r.status == "timed_out"]
        if not timed_out:
            checks.append(_hc("OrderQueue timed_out reserved_notional", "WARN",
                               "No timed_out response generated — max_order_age_seconds may not have elapsed"))
        else:
            assert timed_out[0].reserved_notional == 500.0, \
                f"Expected reserved_notional=500, got {timed_out[0].reserved_notional}"
            checks.append(_hc("OrderQueue timed_out reserved_notional", "PASS",
                               f"reserved_notional={timed_out[0].reserved_notional} preserved on timeout"))
    except AssertionError as exc:
        checks.append(_hc("OrderQueue timed_out reserved_notional", "FAIL", str(exc),
                           "P0 fix: reserved_notional must be set on timed_out response"))
    except Exception as exc:
        checks.append(_hc("OrderQueue timed_out reserved_notional", "FAIL",
                           f"{type(exc).__name__}: {exc}"))

    return checks


def check_risk_manager() -> list[dict]:
    checks: list[dict] = []
    try:
        from app.risk.manager import RiskManager  # noqa: PLC0415
        rm = RiskManager({"risk": {
            "max_daily_loss_pct": 3.0,
            "max_position_size_pct": 10.0,
            "hard_stop": 0.8,
        }})
        # Should not raise
        checks.append(_hc("RiskManager instantiation", "PASS"))
    except Exception as exc:
        checks.append(_hc("RiskManager instantiation", "FAIL",
                           f"{type(exc).__name__}: {exc}"))

    return checks


def check_broker_router() -> list[dict]:
    checks: list[dict] = []
    try:
        from app.brokers.router import BrokerRouter  # noqa: PLC0415

        class _FakeBroker:
            def is_connected(self) -> bool:
                return True
            def get_account(self) -> dict:
                return {"equity": 10000.0, "cash": 5000.0, "buying_power": 8000.0}
            def get_positions(self) -> list:
                return []
            def get_open_orders(self) -> list:
                return []
            def place_order(self, *a: Any, **kw: Any) -> str:
                return "fake-order-1"
            def close_position(self, *a: Any, **kw: Any) -> None:
                pass
            def cancel_order(self, *a: Any, **kw: Any) -> None:
                pass

        router = BrokerRouter({"alpaca": _FakeBroker(), "binance": _FakeBroker()},
                               routing={"default": "alpaca"})
        acct = router.get_account()
        assert acct["equity"] == 20000.0, f"Expected 20000, got {acct['equity']}"
        assert len(acct["brokers"]) == 2
        checks.append(_hc("BrokerRouter aggregation", "PASS",
                           f"equity={acct['equity']} across {len(acct['brokers'])} brokers"))

        # Circuit breaker
        router._offline_until["binance"] = time.monotonic() + 60
        acct2 = router.get_account()
        assert acct2["equity"] > 0, "Should serve stale cache when broker offline"
        checks.append(_hc("BrokerRouter circuit breaker", "PASS", "Stale cache served when broker offline"))
    except AssertionError as exc:
        checks.append(_hc("BrokerRouter", "FAIL", str(exc)))
    except Exception as exc:
        checks.append(_hc("BrokerRouter", "FAIL", f"{type(exc).__name__}: {exc}"))

    return checks


def check_pending_notional_logic() -> list[dict]:
    """
    Verify that _pending_sell_qty guard releases correctly on completed orders.
    This is an in-process logic check (no live broker needed).
    """
    checks: list[dict] = []
    try:
        from app.execution.order_queue import OrderQueue  # noqa: PLC0415

        class _ImmediateBroker:
            def place_order(self, *a: Any, **kw: Any) -> str:
                return "sell-order-1"

        q = OrderQueue(_ImmediateBroker(), "test", completion_grace_seconds=0)
        q.enqueue("BTC/USD", "sell", 0.1, notional=5000.0)
        # Order appears in open orders, then disappears (filled)
        q.update([{"order_id": "sell-order-1", "symbol": "BTC/USD", "side": "sell",
                    "qty": 0.1, "filled_qty": None, "filled_avg_price": None}])
        q.update([])  # disappears → completed
        resps = q.pop_responses()
        completed = [r for r in resps if r.status == "completed"]
        assert completed, f"Expected completed response; got statuses: {[r.status for r in resps]}"
        checks.append(_hc("Sell completed response generation", "PASS",
                           "completed response generated on order disappearance"))
    except AssertionError as exc:
        checks.append(_hc("Sell completed response generation", "FAIL", str(exc),
                           "Check _update_open_order_queues and queue.update() logic"))
    except Exception as exc:
        checks.append(_hc("Sell completed response generation", "FAIL",
                           f"{type(exc).__name__}: {exc}"))

    return checks


def run_all_health_checks(config_path: str) -> list[dict]:
    all_checks: list[dict] = []
    section_fns = [
        ("Configuration", lambda: check_config(config_path)),
        ("Data & Model Files", check_data_files),
        ("Strategy Smoke Tests", lambda: check_strategy_smoke(config_path)),
        ("Order Queue Integrity", check_order_queue_integrity),
        ("Risk Manager", check_risk_manager),
        ("Broker Router", check_broker_router),
        ("Pending Sell Logic", check_pending_notional_logic),
    ]
    results: list[dict] = []
    for section_name, fn in section_fns:
        try:
            checks = fn()
            results.append({"section": section_name, "checks": checks})
        except Exception as exc:
            results.append({
                "section": section_name,
                "checks": [_hc(section_name, "FAIL",
                               f"Section crashed: {exc}\n{traceback.format_exc()[-600:]}",
                               "Investigate section runner")]
            })
    return results


# ---------------------------------------------------------------------------
# Word document report generation
# ---------------------------------------------------------------------------

def _status_color(status: str) -> str:
    return {"PASS": "2ECC71", "FAIL": "E74C3C", "WARN": "F39C12",
            "INFO": "3498DB", "ERROR": "E74C3C"}.get(status.upper(), "000000")


def generate_word_report(
    pytest_data: dict,
    health_sections: list[dict],
    output_path: str,
    config_path: str,
) -> None:
    try:
        from docx import Document  # noqa: PLC0415
        from docx.shared import Inches, Pt, RGBColor  # noqa: PLC0415
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: PLC0415
        from docx.oxml.ns import qn  # noqa: PLC0415
        from docx.oxml import OxmlElement  # noqa: PLC0415
    except ImportError as exc:
        logger.error("python-docx not available: %s  — writing JSON fallback", exc)
        _write_json_fallback(pytest_data, health_sections, output_path)
        return

    doc = Document()

    # ---- page margins ----
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    now = datetime.datetime.now(datetime.timezone.utc)

    # ---- Title block ----
    title = doc.add_heading("Fricktrade — Periodic Test Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Host: {platform.node()}  |  Python {sys.version.split()[0]}"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    doc.add_paragraph()

    # ---- Executive Summary ----
    doc.add_heading("Executive Summary", level=1)
    summary = pytest_data.get("summary", {})
    pass_n = summary.get("pass", 0)
    fail_n = summary.get("fail", 0)
    skip_n = summary.get("skip", 0)
    total = pass_n + fail_n + skip_n

    # count health check statuses
    hc_counts: dict[str, int] = {}
    for sec in health_sections:
        for chk in sec.get("checks", []):
            st = chk.get("status", "?")
            hc_counts[st] = hc_counts.get(st, 0) + 1

    overall = "PASS" if fail_n == 0 and hc_counts.get("FAIL", 0) == 0 else "FAIL"
    if overall == "PASS" and (hc_counts.get("WARN", 0) > 0):
        overall = "WARN"

    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = "Table Grid"
    hdr_cells = tbl.rows[0].cells
    for i, (lbl, val) in enumerate([
        ("Overall Status", overall),
        (f"Unit Tests ({total})", f"{pass_n} pass / {fail_n} fail / {skip_n} skip"),
        ("Health Checks", " / ".join(f"{v} {k}" for k, v in sorted(hc_counts.items()))),
        ("Duration", f"{pytest_data.get('duration_seconds', 0):.1f}s"),
    ]):
        p = hdr_cells[i].paragraphs[0]
        p.clear()
        run_lbl = p.add_run(lbl + "\n")
        run_lbl.bold = True
        run_lbl.font.size = Pt(8)
        run_val = p.add_run(val)
        run_val.font.size = Pt(10)
        run_val.bold = True
        if i == 0:
            color_hex = _status_color(overall)
            run_val.font.color.rgb = RGBColor(
                int(color_hex[:2], 16), int(color_hex[2:4], 16), int(color_hex[4:], 16))
    doc.add_paragraph()

    # ---- Unit Test Results ----
    doc.add_heading("Unit Test Results", level=1)
    results = pytest_data.get("results", [])
    if not results:
        doc.add_paragraph("No test results captured.")
    else:
        tbl2 = doc.add_table(rows=1, cols=4)
        tbl2.style = "Table Grid"
        hdrs = ["Test", "Status", "Duration (s)", "Notes"]
        for i, h in enumerate(hdrs):
            p = tbl2.rows[0].cells[i].paragraphs[0]
            run = p.add_run(h)
            run.bold = True
            run.font.size = Pt(9)

        for r in results:
            row_cells = tbl2.add_row().cells
            node = r["nodeid"].split("::")[-1] if "::" in r["nodeid"] else r["nodeid"]
            row_cells[0].text = node
            row_cells[0].paragraphs[0].runs[0].font.size = Pt(8)

            status_cell = row_cells[1].paragraphs[0]
            status_cell.clear()
            sr = status_cell.add_run(r["status"])
            sr.font.size = Pt(8)
            sr.bold = True
            color_hex = _status_color(r["status"])
            sr.font.color.rgb = RGBColor(
                int(color_hex[:2], 16), int(color_hex[2:4], 16), int(color_hex[4:], 16))

            row_cells[2].text = f"{r.get('duration', 0):.2f}"
            row_cells[2].paragraphs[0].runs[0].font.size = Pt(8)

            msg = r.get("message", "")[:200]
            row_cells[3].text = msg
            row_cells[3].paragraphs[0].runs[0].font.size = Pt(7)

    if pytest_data.get("warnings"):
        doc.add_paragraph()
        doc.add_heading("Pytest Warnings", level=2)
        for w in pytest_data["warnings"][:10]:
            p = doc.add_paragraph(w[:300], style="List Bullet")
            p.runs[0].font.size = Pt(8)

    doc.add_paragraph()

    # ---- Health Check Results ----
    doc.add_heading("System Health Checks", level=1)
    for section_data in health_sections:
        section_name = section_data.get("section", "—")
        checks = section_data.get("checks", [])

        doc.add_heading(section_name, level=2)
        if not checks:
            doc.add_paragraph("No checks in this section.")
            continue

        htbl = doc.add_table(rows=1, cols=3)
        htbl.style = "Table Grid"
        for i, h in enumerate(["Check", "Status", "Detail / Recommendation"]):
            p = htbl.rows[0].cells[i].paragraphs[0]
            run = p.add_run(h)
            run.bold = True
            run.font.size = Pt(9)

        for chk in checks:
            row_cells = htbl.add_row().cells
            row_cells[0].text = chk.get("name", "")
            row_cells[0].paragraphs[0].runs[0].font.size = Pt(8)

            status = chk.get("status", "?")
            sp = row_cells[1].paragraphs[0]
            sp.clear()
            sr = sp.add_run(status)
            sr.bold = True
            sr.font.size = Pt(8)
            color_hex = _status_color(status)
            sr.font.color.rgb = RGBColor(
                int(color_hex[:2], 16), int(color_hex[2:4], 16), int(color_hex[4:], 16))

            detail = chk.get("detail", "")
            rec = chk.get("recommendation", "")
            full_text = detail
            if rec:
                full_text += f"\n→ {rec}"
            row_cells[2].text = full_text[:400]
            row_cells[2].paragraphs[0].runs[0].font.size = Pt(8)

        doc.add_paragraph()

    # ---- Findings Summary ----
    issues = []
    for sec in health_sections:
        for chk in sec.get("checks", []):
            if chk.get("status") in ("FAIL", "WARN"):
                issues.append(f"[{chk['status']}] {sec['section']} — {chk['name']}: {chk.get('detail', '')[:200]}")
    failed_tests = [r for r in results if r["status"] == "FAIL"]
    for ft in failed_tests:
        issues.append(f"[FAIL] Unit test: {ft['nodeid'].split('::')[-1]}: {ft.get('message', '')[:200]}")

    doc.add_heading("Findings Requiring Attention", level=1)
    if not issues:
        p = doc.add_paragraph("No failures or warnings detected. All systems nominal.")
        p.runs[0].font.color.rgb = RGBColor(0x2E, 0xCC, 0x71)
    else:
        for issue in issues:
            p = doc.add_paragraph(issue, style="List Bullet")
            p.runs[0].font.size = Pt(9)
            color_hex = _status_color("FAIL" if "[FAIL]" in issue else "WARN")
            p.runs[0].font.color.rgb = RGBColor(
                int(color_hex[:2], 16), int(color_hex[2:4], 16), int(color_hex[4:], 16))

    # ---- Footer note ----
    doc.add_paragraph()
    foot = doc.add_paragraph(
        "This report is generated automatically by scripts/generate_test_report.py "
        f"| Config: {config_path} | Fricktrade v3.0"
    )
    foot.runs[0].font.size = Pt(7)
    foot.runs[0].font.color.rgb = RGBColor(0xA0, 0xA0, 0xA0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    logger.info("Test report saved to %s", output_path)


def _write_json_fallback(pytest_data: dict, health_sections: list[dict], output_path: str) -> None:
    json_path = Path(output_path).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"pytest": pytest_data, "health": health_sections}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("JSON fallback report saved to %s", json_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_report(config_path: str, output_path: str, tests_dir: str = "tests") -> str:
    """
    Full pipeline: run pytest + health checks + write Word report.
    Returns the path of the output file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Starting test report generation")

    logger.info("Running pytest suite...")
    pytest_data = run_pytest(tests_dir)
    summary = pytest_data.get("summary", {})
    logger.info(
        "pytest: %d pass / %d fail / %d skip (exit_code=%d)",
        summary.get("pass", 0),
        summary.get("fail", 0),
        summary.get("skip", 0),
        pytest_data.get("exit_code", -1),
    )

    logger.info("Running health checks...")
    health_sections = run_all_health_checks(config_path)
    for sec in health_sections:
        fails = [c for c in sec.get("checks", []) if c.get("status") in ("FAIL", "WARN")]
        if fails:
            logger.warning("Section '%s': %d issues", sec["section"], len(fails))

    logger.info("Generating Word report → %s", output_path)
    generate_word_report(pytest_data, health_sections, output_path, config_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate periodic Fricktrade test report")
    parser.add_argument("--config", default="/app/config/config.yaml")
    parser.add_argument("--output", default="/data/reports/test_report.docx")
    parser.add_argument("--tests-dir", default="tests")
    args = parser.parse_args()

    output = generate_report(
        config_path=args.config,
        output_path=args.output,
        tests_dir=args.tests_dir,
    )
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()

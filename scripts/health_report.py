# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Periodic system health reporter.

Runs every `window_hours` hours (default 6). Collects data from:
  - Prometheus metrics endpoint (strategy win rates, account, rejections)
  - Docker container logs for ALL services (failures, errors, warnings)
  - Decision trace JSONL files (executions and skips within window)
  - Trader checkpoint (active positions, dust)
  - Meta-orchestrator result files (health grade, changes)

Writes to /data/reports/health/health_YYYY-MM-DD_HH.json + .txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

try:
    import docker as _docker_sdk
except ImportError:
    _docker_sdk = None  # type: ignore

from app.utils.config import load_config

logger = logging.getLogger("health_reporter")

# Rejection reasons to scan for in container logs
_REJECTION_REASONS = [
    "floors_to_zero",
    "not_fractionable",
    "insufficient_stablecoin",
    "min_order_notional",
    "insufficient_cash",
    "pdt_protection",
    "exposure_cap",
    "pending_leverage_cap",
    "stuck_cooldown",
    "symbol_error",
]

# Services that produce meaningful failure lines (avoid noisy infra containers)
_MONITORED_SERVICES = {
    "trader", "healthwatch", "learner", "daily-report",
    "tests-when-closed", "market-cache", "api", "calendar-updater",
}

# Log lines to skip even at ERROR/WARNING level (known noise)
_NOISE_PATTERNS = [
    "yfinance",
    "AFC is enabled",
    "no price data available",
]


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------

def _collect_prometheus(url: str) -> dict:
    try:
        raw = urlopen(url, timeout=10).read().decode()
        return _parse_prometheus(raw)
    except Exception as exc:
        logger.warning("Prometheus unreachable: %s", exc)
        return {}


def _parse_prometheus(text: str) -> dict[str, list[tuple]]:
    result: dict[str, list] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'(\w+)(\{[^}]*\})?\s+([\d.e+\-NaI]+)', line)
        if not m:
            continue
        name, labels_str, value_str = m.group(1), m.group(2) or "", m.group(3)
        labels = dict(re.findall(r'(\w+)="([^"]*)"', labels_str))
        try:
            value = float(value_str)
        except ValueError:
            continue
        result.setdefault(name, []).append((labels, value))
    return result


def _prom_get(prom: dict, metric: str, label_filter: dict | None = None) -> list[tuple]:
    rows = prom.get(metric, [])
    if not label_filter:
        return rows
    return [(lbl, val) for lbl, val in rows if all(lbl.get(k) == v for k, v in label_filter.items())]


# ---------------------------------------------------------------------------
# Docker logs helpers
# ---------------------------------------------------------------------------

def _get_docker_client(docker_host: str):
    if _docker_sdk is None:
        return None
    host = docker_host or os.environ.get("DOCKER_HOST", "unix://var/run/docker.sock")
    try:
        return _docker_sdk.DockerClient(base_url=host)
    except Exception as exc:
        logger.warning("Docker client init failed: %s", exc)
        return None


def _project_containers(client, project: str) -> list:
    label = f"com.docker.compose.project={project}"
    return client.containers.list(all=True, filters={"label": label})


def _is_noise(line: str) -> bool:
    for pat in _NOISE_PATTERNS:
        if pat in line:
            return True
    return False


def _collect_all_container_logs(client, project: str, window_start: datetime) -> dict:
    """
    Collect failures from ALL monitored containers.
    Returns per-service dict with error_count, warning_count, failure_lines list.
    """
    since_ts = int(window_start.timestamp())
    result: dict[str, dict] = {}

    if client is None:
        return result

    try:
        containers = _project_containers(client, project)
    except Exception as exc:
        logger.warning("Could not list containers: %s", exc)
        return result

    for container in containers:
        service = container.labels.get("com.docker.compose.service", "")
        if service not in _MONITORED_SERVICES:
            continue
        try:
            raw = container.logs(since=since_ts, timestamps=False).decode(errors="replace")
        except Exception as exc:
            logger.debug("Could not get logs for %s: %s", service, exc)
            continue

        errors = 0
        warnings = 0
        failures: list[str] = []

        for line in raw.splitlines():
            if _is_noise(line):
                continue
            is_error = " ERROR " in line
            is_warning = " WARNING " in line
            if is_error:
                errors += 1
                failures.append(line.strip()[-300:])
            elif is_warning:
                warnings += 1
                # Only record warnings that look like actual failures
                if any(kw in line for kw in (
                    "failed", "failure", "error", "exception", "rejected",
                    "timeout", "timed_out", "crash", "critical", "blocked"
                )):
                    failures.append(line.strip()[-300:])

        if errors > 0 or warnings > 0 or failures:
            result[service] = {
                "error_count": errors,
                "warning_count": warnings,
                "notable_failures": failures[-50:],  # cap to last 50
            }

    return result


def _collect_trader_logs(client, project: str, window_start: datetime) -> dict:
    """Collect trader-specific metrics: LLM cost, rejection counts, trade notional."""
    since_ts = int(window_start.timestamp())
    result = {
        "llm_cost_daily_usd": None,
        "llm_calls": 0,
        "rejection_counts": {},
        "trade_notional_usd": 0.0,
        "trades_executed": 0,
    }
    if client is None:
        return result

    trader_container = None
    try:
        for c in _project_containers(client, project):
            if c.labels.get("com.docker.compose.service") == "trader":
                trader_container = c
                break
    except Exception:
        return result

    if trader_container is None:
        return result

    try:
        raw = trader_container.logs(since=since_ts, timestamps=False).decode(errors="replace")
    except Exception as exc:
        logger.warning("Could not fetch trader logs: %s", exc)
        return result

    cost_re = re.compile(r'\(daily: \$([\d.]+)\)')
    trade_re = re.compile(r'"event":\s*"trade_executed".*?"notional":\s*([\d.e+\-]+)')

    for line in raw.splitlines():
        # LLM cost — take max (it's cumulative within the day)
        m = cost_re.search(line)
        if m:
            val = float(m.group(1))
            result["llm_cost_daily_usd"] = max(result["llm_cost_daily_usd"] or 0.0, val)
            result["llm_calls"] += 1

        # Trade notional
        m2 = trade_re.search(line)
        if m2:
            result["trade_notional_usd"] += float(m2.group(1))
            result["trades_executed"] += 1

        # Rejection reasons (window-scoped)
        for reason in _REJECTION_REASONS:
            if reason in line and ("rejected" in line.lower() or "failed" in line.lower()
                                   or "blocked" in line.lower() or "skip" in line.lower()):
                result["rejection_counts"][reason] = result["rejection_counts"].get(reason, 0) + 1

    return result


# ---------------------------------------------------------------------------
# Decision trace
# ---------------------------------------------------------------------------

def _collect_trace(trace_dir: str, window_start: datetime) -> dict:
    buys: dict[str, int] = {}
    sells: dict[str, int] = {}
    skips: dict[str, int] = {}

    for date in sorted({window_start.date(), datetime.now(timezone.utc).date()}):
        path = Path(trace_dir) / f"{date}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ts_str = r.get("ts", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < window_start:
                        continue
                except Exception:
                    continue
                broker = r.get("broker", "unknown")
                decision = r.get("decision", "")
                if decision == "order_enqueued":
                    action = r.get("action", "")
                    if action == "buy":
                        buys[broker] = buys.get(broker, 0) + 1
                    elif action in ("sell", "exit"):
                        sells[broker] = sells.get(broker, 0) + 1
                elif decision == "skip":
                    reason = r.get("reason", "unknown")
                    skips[reason] = skips.get(reason, 0) + 1

    return {"buys_by_broker": buys, "sells_by_broker": sells, "skip_reasons": skips}


# ---------------------------------------------------------------------------
# Checkpoint (positions)
# ---------------------------------------------------------------------------

def _collect_checkpoint(checkpoint_path: str) -> dict:
    try:
        data = json.loads(Path(checkpoint_path).read_text())
    except Exception:
        return {"positions": {}, "dust_positions": {}, "dust_count": 0}

    payload = data.get("payload", {})
    broker_states = payload.get("broker_states", {})
    positions: dict[str, list] = {}
    dust: dict[str, list] = {}

    for bname, bstate in broker_states.items():
        pos_state = bstate.get("position_state", {})
        real = []
        dust_syms = []
        for sym, ps in pos_state.items():
            qty = float(ps.get("qty", 0) or 0)
            if qty < 1e-6:
                dust_syms.append(sym)
            else:
                real.append({
                    "symbol": sym,
                    "qty": qty,
                    "avg_entry": ps.get("avg_entry"),
                    "strategy": ps.get("strategy"),
                    "opened_at": ps.get("opened_at"),
                })
        positions[bname] = real
        dust[bname] = dust_syms

    dust_count = sum(len(v) for v in dust.values())
    return {"positions": positions, "dust_positions": dust, "dust_count": dust_count}


# ---------------------------------------------------------------------------
# Meta-orchestrator
# ---------------------------------------------------------------------------

def _collect_meta_orch(meta_orch_dir: str, window_start: datetime) -> dict:
    d = Path(meta_orch_dir)
    result: dict = {}

    lr = d / "last_result.json"
    if lr.exists():
        try:
            r = json.loads(lr.read_text())
            result["health_grade"] = r.get("health_grade")
            result["primary_finding"] = r.get("primary_finding", "")
            result["anomalies"] = r.get("anomalies", [])
            result["proposed_changes_count"] = len(r.get("proposed_config_changes", []))
        except Exception:
            pass

    changes = []
    cl = d / "changes.jsonl"
    if cl.exists():
        try:
            with open(cl) as f:
                for line in f:
                    try:
                        c = json.loads(line)
                        ts_str = c.get("ts", "")
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts >= window_start:
                            changes.append({
                                "parameter": c.get("parameter"),
                                "old_value": c.get("old_value"),
                                "new_value": c.get("new_value"),
                            })
                    except Exception:
                        continue
        except Exception:
            pass

    result["changes_in_window"] = changes
    return result


# ---------------------------------------------------------------------------
# Prometheus-derived summaries
# ---------------------------------------------------------------------------

def _extract_strategy_win_rates(prom: dict) -> dict:
    rates: dict[str, dict] = {}
    for labels, value in prom.get("strategy_win_rate", []):
        rates.setdefault(labels.get("strategy", ""), {})["win_rate"] = round(value, 3)
    for labels, value in prom.get("strategy_avg_pnl_pct", []):
        rates.setdefault(labels.get("strategy", ""), {})["avg_pnl_pct"] = round(value, 3)
    for labels, value in prom.get("strategy_trades_realized_total", []):
        rates.setdefault(labels.get("strategy", ""), {})["trades"] = int(value)
    for labels, value in prom.get("strategy_disabled", []):
        if value > 0:
            rates.setdefault(labels.get("strategy", ""), {})["disabled"] = True
    return {k: v for k, v in rates.items() if k}


def _extract_account(prom: dict) -> dict:
    account: dict[str, dict] = {}
    for labels, value in prom.get("account_total_by_broker", []):
        account.setdefault(labels.get("broker", ""), {})["equity"] = round(value, 2)
    for labels, value in prom.get("account_cash_by_broker", []):
        account.setdefault(labels.get("broker", ""), {})["cash"] = round(value, 2)
    return account


def _extract_prom_rejections(prom: dict) -> dict:
    rejects: dict[str, int] = {}
    for labels, value in prom.get("order_rejects_total", []):
        reason = labels.get("reason", "other")
        rejects[reason] = rejects.get(reason, 0) + int(value)
    return rejects


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

def _format_text_summary(report: dict) -> str:
    lines = [
        "=" * 60,
        "  FRICKTRADE SYSTEM HEALTH REPORT",
        f"  Generated : {report['generated_at']}",
        f"  Window    : last {report['window_hours']}h  (from {report['window_start']})",
        "=" * 60,
        "",
        "--- TRADE EXECUTIONS (this window) ---",
    ]
    execs = report.get("trade_executions", {})
    if execs:
        for broker, counts in sorted(execs.items()):
            lines.append(f"  {broker}: buy={counts.get('buy', 0)}  sell={counts.get('sell', 0)}")
    else:
        lines.append("  (none)")

    notional = report.get("trade_notional_usd", 0.0)
    lines.append(f"  Total notional: ${notional:,.2f}")

    lines += ["", "--- ORDER SKIPS IN WINDOW (top reasons) ---"]
    skips = sorted(report.get("skip_reasons_window", {}).items(), key=lambda x: -x[1])
    for reason, count in skips[:10]:
        lines.append(f"  {reason}: {count}")
    if not skips:
        lines.append("  (none)")

    lines += ["", "--- ORDER REJECTIONS IN WINDOW (from logs) ---"]
    rej_w = sorted(report.get("order_rejections_window", {}).items(), key=lambda x: -x[1])
    for reason, count in rej_w:
        lines.append(f"  {reason}: {count}")
    if not rej_w:
        lines.append("  (none)")

    lines += ["", f"--- DUST POSITIONS: {report.get('dust_count', 0)} total ---"]
    for broker, syms in report.get("dust_positions", {}).items():
        if syms:
            lines.append(f"  {broker}: {', '.join(syms)}")

    lines += ["", "--- STRATEGY WIN RATES (cumulative) ---"]
    for strat, stats in sorted(report.get("strategy_win_rates", {}).items()):
        disabled = " [DISABLED]" if stats.get("disabled") else ""
        lines.append(
            f"  {strat}{disabled}: "
            f"win={stats.get('win_rate', 0):.1%}  "
            f"avg_pnl={stats.get('avg_pnl_pct', 0):+.2f}%  "
            f"n={stats.get('trades', 0)}"
        )

    lines += ["", "--- ACCOUNT EQUITY ---"]
    for broker, vals in sorted(report.get("account", {}).items()):
        equity = vals.get("equity", 0)
        cash = vals.get("cash", 0)
        lines.append(f"  {broker}: equity=${equity:,.2f}  cash=${cash:,.2f}")

    llm_cost = report.get("llm_cost_daily_usd")
    cost_str = f"${llm_cost:.4f}" if llm_cost is not None else "n/a"
    lines += [
        "", "--- LLM USAGE ---",
        f"  Daily cost (cumulative): {cost_str}",
        f"  LLM calls in window: {report.get('llm_calls_window', 0)}",
    ]

    mo = report.get("meta_orch", {})
    lines += [
        "", "--- META-ORCHESTRATOR ---",
        f"  Health grade   : {mo.get('health_grade', 'n/a')}",
        f"  Primary finding: {mo.get('primary_finding', 'n/a')}",
        f"  Config changes applied in window: {len(mo.get('changes_in_window', []))}",
    ]
    for ch in mo.get("changes_in_window", []):
        lines.append(f"    {ch['parameter']}: {ch['old_value']} → {ch['new_value']}")

    lines += ["", "--- CONTAINER FAILURES (all services, this window) ---"]
    container_failures = report.get("container_failures", {})
    total_errors = sum(v.get("error_count", 0) for v in container_failures.values())
    total_warnings = sum(v.get("warning_count", 0) for v in container_failures.values())
    lines.append(f"  Total ERROR lines  : {total_errors}")
    lines.append(f"  Total WARNING lines: {total_warnings}")
    if container_failures:
        for service, data in sorted(container_failures.items()):
            lines.append(
                f"  [{service}] errors={data['error_count']} warnings={data['warning_count']}"
            )
            for fl in data.get("notable_failures", [])[-5:]:
                lines.append(f"    >> {fl[:200]}")
    else:
        lines.append("  (no failures detected)")

    lines += ["", "--- ACTIVE POSITIONS ---"]
    for broker, pos_list in sorted(report.get("active_positions", {}).items()):
        real = [p for p in pos_list if float(p.get("qty", 0) or 0) >= 1e-6]
        lines.append(f"  {broker} ({len(real)} positions):")
        for p in real[:15]:
            lines.append(
                f"    {p['symbol']}  qty={p['qty']:.6f}  "
                f"entry={p.get('avg_entry', '?')}  strat={p.get('strategy', '?')}"
            )
    lines += ["", "=" * 60]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main collection + report assembly
# ---------------------------------------------------------------------------

def run_once(cfg: dict, window_hours: int, output_dir: Path) -> dict:
    hr_cfg = cfg.get("health_reporter", {}) or {}
    trace_dir = hr_cfg.get("decision_trace_dir", "/data/reports/decision_trace")
    checkpoint_path = hr_cfg.get("checkpoint_path", "/data/checkpoints/trader.json")
    meta_orch_dir = hr_cfg.get("meta_orch_dir", "/data/reports/meta_orch")
    metrics_url = hr_cfg.get("trader_metrics_url", "http://trader:8001/metrics")
    docker_host = hr_cfg.get("docker_host", "") or os.environ.get("DOCKER_HOST", "")
    compose_project = hr_cfg.get("compose_project", "fricktrade")

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=window_hours)

    logger.info("Collecting health report (window=%dh, from %s)…", window_hours, window_start.isoformat())

    prom = _collect_prometheus(metrics_url)
    docker_client = _get_docker_client(docker_host)

    container_failures = _collect_all_container_logs(docker_client, compose_project, window_start)
    trader_logs = _collect_trader_logs(docker_client, compose_project, window_start)
    trace = _collect_trace(trace_dir, window_start)
    chk = _collect_checkpoint(checkpoint_path)
    mo = _collect_meta_orch(meta_orch_dir, window_start)

    all_brokers = set(list(trace["buys_by_broker"]) + list(trace["sells_by_broker"]))
    trade_executions = {
        b: {
            "buy": trace["buys_by_broker"].get(b, 0),
            "sell": trace["sells_by_broker"].get(b, 0),
        }
        for b in sorted(all_brokers)
    }

    report = {
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
        "window_start": window_start.isoformat(),
        "trade_executions": trade_executions,
        "trade_notional_usd": round(trader_logs["trade_notional_usd"], 2),
        "trades_executed_window": trader_logs["trades_executed"],
        "skip_reasons_window": dict(sorted(trace["skip_reasons"].items(), key=lambda x: -x[1])),
        "order_rejections_cumulative": _extract_prom_rejections(prom),
        "order_rejections_window": trader_logs["rejection_counts"],
        "dust_positions": chk["dust_positions"],
        "dust_count": chk["dust_count"],
        "strategy_win_rates": _extract_strategy_win_rates(prom),
        "llm_cost_daily_usd": trader_logs["llm_cost_daily_usd"],
        "llm_calls_window": trader_logs["llm_calls"],
        "container_failures": container_failures,
        "meta_orch": mo,
        "active_positions": chk["positions"],
        "account": _extract_account(prom),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = now.strftime("%Y-%m-%d_%H")
    json_path = output_dir / f"health_{ts_tag}.json"
    txt_path = output_dir / f"health_{ts_tag}.txt"

    json_path.write_text(json.dumps(report, indent=2, default=str))
    summary = _format_text_summary(report)
    txt_path.write_text(summary)

    logger.info("Health report written → %s\n%s", json_path, summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fricktrade periodic health reporter")
    parser.add_argument("--config", default="/app/config/config.yaml")
    parser.add_argument("--window-hours", type=int, default=6)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    while True:
        cfg = load_config(args.config)
        hr_cfg = cfg.get("health_reporter", {}) or {}
        if hr_cfg.get("enabled", True):
            output_dir = Path(args.output_dir or hr_cfg.get("output_dir", "/data/reports/health"))
            window_hours = args.window_hours or int(hr_cfg.get("window_hours", 6))
            try:
                run_once(cfg, window_hours, output_dir)
            except Exception:
                logger.exception("Health report run failed")
        else:
            logger.info("Health reporter disabled in config — sleeping")

        if args.once:
            break
        interval = int(hr_cfg.get("window_hours", args.window_hours)) * 3600
        logger.info("Next health report in %d seconds", interval)
        time.sleep(interval)


if __name__ == "__main__":
    main()

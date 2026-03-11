# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Strategic Meta Orchestrator — weekly (or emergency) config rebalancer.

Runs in two conditions:
  1. Weekly — Sunday at a configurable UTC hour, after ≥3 session reports exist.
  2. Emergency — when the last N consecutive session grades are D or F.

Reads:
  - Accumulated PostSession reports from session_reports_dir (last 7 days)
  - Tactical Meta Orchestrator change log from tactical_changes_path
  - Current strategy weights and config snapshot from the live trader

Produces a strategic_baseline.json consumed by TacticalMetaOrchestrator,
which constrains its weight moves to ±corridor_pct of these baselines.

Change application uses a 1-hour delay (longer than the tactical layer's
5 minutes) and auto-applies unless strategic_auto_apply is False.

This is the slow, deliberate counterpart to TacticalMetaOrchestrator.
Together they form a two-tier hierarchy:
  Strategic (weekly)  → sets weight baselines + parameter corridors
  Tactical  (15-min)  → adjusts within ±corridor_pct of those baselines

The file is a complete rewrite of the original prototype (which was never
wired into the trading loop and knew only 4 of the 10 active strategies).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from prometheus_client import Counter, Gauge
    _STRATEGIC_CYCLES = Counter("strategic_orch_cycles_total", "Strategic review cycles fired")
    _STRATEGIC_GRADE  = Gauge("strategic_orch_confidence", "Confidence of last strategic review (0-1)")
    _STRATEGIC_WEIGHT = Gauge("strategic_orch_baseline_weight", "Baseline weight per strategy", ["strategy"])
    _PROMETHEUS_OK = True
except Exception:
    _PROMETHEUS_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All active strategies (must match config.yaml strategy.names)
# ---------------------------------------------------------------------------

_STRATEGY_DESCRIPTIONS = {
    "trend_following":      "equity momentum — long trending stocks across sectors",
    "factor_model":         "cross-sectional equity — momentum/value/quality factors; EXCLUDED from crypto",
    "pattern_trading":      "chart pattern recognition — works with clear formations",
    "stat_arb_pairs":       "mean-reversion pairs — range-bound markets",
    "top_movers_rf":        "return-ranker RF model — top daily movers ranked by ML score",
    "crypto_momentum":      "crypto trend-following — directional breakouts with volume confirmation",
    "crypto_mean_reversion":"crypto range-bound — RSI-based reversals in sideways crypto",
    "gap_reversal":         "equity gap fill — fade overnight gaps at open",
    "earnings_drift":       "PEAD — post-earnings announcement drift in equities",
    "rl_policy":            "PPO reinforcement learning — symbol-agnostic price/volume policy",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the STRATEGIC ORCHESTRATOR for an autonomous crypto and equity trading system.
You run weekly (or when triggered by consecutive losing sessions) and make deliberate,
evidence-based adjustments to strategy weight baselines and system parameters.

Your role in the two-tier hierarchy:
  YOU (strategic, weekly):  set weight BASELINES and the corridors within which
                            the 15-minute Tactical Orchestrator is allowed to move.
  Tactical (15-min):        adjusts within ±corridor_pct of your baselines in real-time.

ACTIVE STRATEGIES (10 total):
{strategy_block}

DECISION FRAMEWORK:
1. WEEK-LEVEL EVIDENCE: look at win rates, P&L, and regime fit across the FULL week.
   Do not react to single sessions. Require ≥5 fills per strategy before adjusting weight.
2. REGIME FORECAST: what macro regime do you expect next week? Align weights accordingly.
3. STRUCTURAL ISSUES: if a strategy has been consistently broken (not just unlucky),
   reduce its weight sharply — do not wait for gradual decay.
4. TACTICAL DRIFT: review what the Tactical Orchestrator changed this week. If it
   consistently pushed a weight in one direction, your baseline may need to follow.
5. CORRIDOR SIZING: wider corridors (±40%) for strategies with high conviction;
   narrower (±15%) for strategies with thin or volatile evidence.

HARD CONSTRAINTS:
- Strategy weights must sum to 1.0.
- No weight below 0.03 or above 0.45.
- No weight may move more than 20% from its current baseline in a single review.
- Parameter changes: max 25% from current value.
- Never change: broker credentials, API keys, or any parameter not in the bounds table.
- If evidence is insufficient (< 3 reports), produce a conservative review that
  changes nothing material and says so clearly.

Respond ONLY with valid JSON — no markdown, no text outside the JSON.\
"""


def _build_user_prompt(
    session_reports: list[dict],
    tactical_changes: list[dict],
    current_weights: dict[str, float],
    current_cfg_snapshot: dict,
    trigger_reason: str,
    bounds: dict,
) -> str:
    strategy_block = "\n".join(
        f"  {name:<28} — {desc}"
        for name, desc in _STRATEGY_DESCRIPTIONS.items()
    )
    system_prompt = _SYSTEM_PROMPT.format(strategy_block=strategy_block)

    # Session summary
    session_lines = []
    for r in session_reports:
        grade = r.get("overall_grade", "?")
        date  = r.get("date", "?")
        findings = r.get("key_findings") or []
        top = findings[0].get("finding", "—")[:80] if findings else "—"
        strat_grades = {
            a.get("strategy", "?"): a.get("grade", "?")
            for a in (r.get("strategy_assessments") or [])
        }
        sg = " | ".join(f"{s}:{g}" for s, g in list(strat_grades.items())[:6])
        session_lines.append(f"  {date}  Grade:{grade}  Strategies:[{sg}]  Top finding: {top}")
    sessions_str = "\n".join(session_lines) if session_lines else "  (none available)"

    # Strategy cross-session performance
    strat_agg: dict[str, dict] = {}
    for r in session_reports:
        for a in (r.get("strategy_assessments") or []):
            name = a.get("strategy", "unknown")
            if name not in strat_agg:
                strat_agg[name] = {"grades": [], "pnl": 0.0, "trades": 0,
                                   "adjust_votes": {"increase": 0, "decrease": 0, "maintain": 0}}
            strat_agg[name]["grades"].append(a.get("grade", "?"))
            strat_agg[name]["pnl"] += float(a.get("pnl", 0) or 0)
            strat_agg[name]["trades"] += int(a.get("trades", 0) or 0)
            adj = str(a.get("should_adjust_weight", "maintain") or "maintain")
            strat_agg[name]["adjust_votes"][adj] = strat_agg[name]["adjust_votes"].get(adj, 0) + 1

    strat_lines = []
    for name in _STRATEGY_DESCRIPTIONS:
        cur_w = current_weights.get(name, 0.0)
        if name in strat_agg:
            d = strat_agg[name]
            vote = max(d["adjust_votes"], key=d["adjust_votes"].get)
            strat_lines.append(
                f"  {name:<28} weight={cur_w:.3f}  grades={','.join(d['grades'])}"
                f"  trades={d['trades']}  pnl={d['pnl']:+.2f}  post-session-vote={vote}"
            )
        else:
            strat_lines.append(
                f"  {name:<28} weight={cur_w:.3f}  (no session data)"
            )
    strat_str = "\n".join(strat_lines)

    # Tactical changes summary
    if tactical_changes:
        tac_lines = []
        param_freq: dict[str, int] = {}
        for ch in tactical_changes:
            p = ch.get("parameter", "?")
            param_freq[p] = param_freq.get(p, 0) + 1
        for param, count in sorted(param_freq.items(), key=lambda x: -x[1])[:12]:
            last = next((c for c in reversed(tactical_changes) if c.get("parameter") == param), {})
            tac_lines.append(
                f"  {param:<52} changed {count}x  last: {last.get('old_value','?')} → {last.get('new_value','?')}"
            )
        tac_str = "\n".join(tac_lines)
    else:
        tac_str = "  (no tactical changes recorded yet)"

    # Bounds table
    bounds_lines = []
    for k, v in sorted(bounds.items()):
        if k.startswith("strategy_weight"):
            continue
        cur = _get_nested(current_cfg_snapshot, k.split("."))
        bounds_lines.append(f"  {k:<52} [{v[0]}, {v[1]}]  current: {cur}")
    bounds_str = "\n".join(bounds_lines)

    week_start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_reports  = len(session_reports)
    weights_json = json.dumps(current_weights, indent=4)
    corridor_pct = bounds.get("tactical_corridor_pct", 30)

    return f"""\
STRATEGIC REVIEW — {week_start} to {week_end}
Trigger: {trigger_reason}  |  Session reports available: {n_reports}
{'━'*65}

── DAILY SESSION GRADES (most recent first) ────────────────────
{sessions_str}

── STRATEGY CROSS-SESSION PERFORMANCE ─────────────────────────
  strategy                       weight   grades   trades  pnl     post-session-vote
{strat_str}

── TACTICAL ORCHESTRATOR ACTIVITY (this week) ──────────────────
Parameters changed most frequently (shows tactical drift direction):
{tac_str}

── CURRENT STRATEGY WEIGHTS ────────────────────────────────────
{weights_json}

── SAFE PARAMETER BOUNDS ───────────────────────────────────────
{bounds_str}

── TACTICAL CORRIDOR ───────────────────────────────────────────
The Tactical Orchestrator may move each strategy weight ±{corridor_pct}% of the
baseline you set. Example: if you set crypto_momentum=0.13, tactical range is
[{0.13*(1-corridor_pct/100):.3f}, {0.13*(1+corridor_pct/100):.3f}].
Set wider corridors (up to ±50%) for high-conviction strategies, narrower
(±15%) for strategies with thin evidence or high volatility in grades.

── RESPONSE FORMAT ─────────────────────────────────────────────
{{
  "week": "{week_start}",
  "trigger": "{trigger_reason}",
  "evidence_quality": "<sufficient|thin|insufficient>",
  "market_assessment": "<paragraph on this week's regime and market character>",
  "regime_forecast": "<expected regime next week with reasoning>",
  "new_baseline_weights": {{
    // ALL 10 strategies; must sum to 1.0; max ±20% move from current
    "trend_following": <float>,
    "factor_model": <float>,
    "pattern_trading": <float>,
    "stat_arb_pairs": <float>,
    "top_movers_rf": <float>,
    "crypto_momentum": <float>,
    "crypto_mean_reversion": <float>,
    "gap_reversal": <float>,
    "earnings_drift": <float>,
    "rl_policy": <float>
  }},
  "weight_corridors": {{
    // corridor_pct per strategy (15-50); determines tactical movement range
    "trend_following": <int>,
    "factor_model": <int>,
    ...
  }},
  "weight_rationale": "<paragraph justifying the weight decisions>",
  "parameter_changes": [
    {{
      "parameter": "<exact dotted config path>",
      "current_value": <current>,
      "proposed_value": <new>,
      "rationale": "<one sentence>"
    }}
  ],
  "tactical_drift_assessment": "<did the tactical orchestrator move in the right direction this week?>",
  "strategy_notes": [
    {{"strategy": "<name>", "note": "<specific observation or structural issue>"}}
  ],
  "confidence": <float 0.0 to 1.0>,
  "no_change_reason": "<if new_baseline_weights unchanged, explain why>"
}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nested(d: dict, keys: list[str]) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _set_nested(d: dict, keys: list[str], value: Any) -> bool:
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            return False
        d = d[k]
    if keys[-1] not in d:
        return False
    d[keys[-1]] = value
    return True


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Clamp each weight to [0.03, 0.45] and renormalise to sum=1."""
    clamped = {k: max(0.03, min(0.45, float(v))) for k, v in weights.items()}
    total = sum(clamped.values())
    if total <= 0:
        n = len(clamped)
        return {k: 1.0 / n for k in clamped}
    return {k: round(v / total, 6) for k, v in clamped.items()}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StrategicOrchestrator:
    """Weekly (or emergency) strategic config rebalancer."""

    def __init__(self, llm_client: Any, cfg: dict) -> None:
        self._llm = llm_client
        self._cfg = cfg
        so_cfg = cfg.get("strategic_meta_orchestrator", {}) or {}
        self._enabled              = bool(so_cfg.get("enabled", True))
        self._weekly_day           = int(so_cfg.get("weekly_day", 6))   # 0=Mon … 6=Sun
        self._weekly_hour_utc      = int(so_cfg.get("weekly_hour_utc", 6))
        self._emergency_threshold  = int(so_cfg.get("emergency_threshold", 3))
        self._min_reports          = int(so_cfg.get("min_reports", 3))
        self._apply_delay_sec      = int(so_cfg.get("apply_delay_hours", 1)) * 3600
        self._auto_apply           = bool(so_cfg.get("auto_apply", True))
        self._backend              = str(so_cfg.get("backend", "gemini"))
        self._model                = so_cfg.get("model") or None
        self._session_dir          = str(so_cfg.get("session_reports_dir",
                                        "/data/reports/session"))
        self._output_dir           = str(so_cfg.get("output_dir",
                                        "/data/reports/meta_orch"))
        self._tactical_changes_log = str(so_cfg.get("tactical_changes_log",
                                        "/data/reports/meta_orch/changes.jsonl"))
        self._bounds: dict         = dict(so_cfg.get("bounds", {}))
        # Runtime state
        self._last_run: datetime | None = None
        self._pending: list[dict]       = []
        self._applied_log: list[dict]   = []
        self._last_result: dict         = {}
        self._baseline: dict            = {}   # current baseline written to disk
        self._lock = threading.Lock()
        # Reference to trader's live _config_strategy_weights (set externally)
        self._weights_ref: dict[str, float] | None = None
        # Load any existing baseline from disk
        self._load_baseline()
        logger.info(
            "StrategicOrchestrator initialised (weekly=day%d@%02dUTC, emergency=%d consecutive D/F)",
            self._weekly_day, self._weekly_hour_utc, self._emergency_threshold,
        )

    def set_weights_ref(self, weights: dict[str, float]) -> None:
        self._weights_ref = weights

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def maybe_run(self) -> None:
        """Called from trader main loop. Fires a review cycle when appropriate."""
        if not self._enabled:
            return
        now = datetime.now(timezone.utc)
        self._apply_pending(now)
        if self._should_run(now):
            self._last_run = now
            t = threading.Thread(target=self._run_cycle, args=(now,), daemon=True)
            t.start()

    # ------------------------------------------------------------------
    # Scheduling logic
    # ------------------------------------------------------------------

    def _should_run(self, now: datetime) -> bool:
        # Never run if already ran recently (guard against repeated triggers)
        if self._last_run and (now - self._last_run).total_seconds() < 3600 * 6:
            return False
        reports = self._load_session_reports()
        if len(reports) < self._min_reports:
            return False
        # Weekly trigger: correct day + hour, hasn't run this week
        if (now.weekday() == self._weekly_day and now.hour == self._weekly_hour_utc
                and (self._last_run is None
                     or (now - self._last_run).total_seconds() > 3600 * 24 * 6)):
            logger.info("StrategicOrch: weekly trigger (day=%d hour=%d UTC)", self._weekly_day, self._weekly_hour_utc)
            return True
        # Emergency trigger: last N consecutive grades are D or F
        recent = sorted(reports, key=lambda r: r.get("date", ""), reverse=True)
        if len(recent) >= self._emergency_threshold:
            bad = all(
                r.get("overall_grade", "A") in ("D", "F")
                for r in recent[:self._emergency_threshold]
            )
            if bad:
                logger.warning(
                    "StrategicOrch: emergency trigger — %d consecutive D/F sessions",
                    self._emergency_threshold,
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Review cycle
    # ------------------------------------------------------------------

    def _run_cycle(self, cycle_start: datetime) -> None:
        trigger = self._detect_trigger(cycle_start)
        try:
            reports  = self._load_session_reports()
            tac_chgs = self._load_tactical_changes()
            cur_weights = dict(self._weights_ref) if self._weights_ref else {}
            prompt = _build_user_prompt(
                session_reports=reports,
                tactical_changes=tac_chgs,
                current_weights=cur_weights,
                current_cfg_snapshot=self._cfg,
                trigger_reason=trigger,
                bounds=self._bounds,
            )
            strategy_block = "\n".join(
                f"  {name:<28} — {desc}"
                for name, desc in _STRATEGY_DESCRIPTIONS.items()
            )
            system = _SYSTEM_PROMPT.format(strategy_block=strategy_block)
            t0 = time.monotonic()
            response = self._llm.complete(
                backend=self._backend,
                system_prompt=system,
                user_prompt=prompt,
                model=self._model,
                max_tokens=2048,
            )
            latency = time.monotonic() - t0
            parsed = self._parse_response(response.content)
            if parsed:
                with self._lock:
                    self._last_result = parsed
                self._write_baseline(parsed, cur_weights)
                self._queue_changes(parsed, datetime.now(timezone.utc))
                self._save_result(parsed)
                if _PROMETHEUS_OK:
                    _STRATEGIC_CYCLES.inc()
                    _STRATEGIC_GRADE.set(float(parsed.get("confidence", 0.5)))
                logger.info(
                    "StrategicOrch review done in %.1fs: confidence=%.2f  changes=%d  trigger=%s",
                    latency, parsed.get("confidence", 0), len(parsed.get("parameter_changes") or []),
                    trigger,
                )
        except Exception as exc:
            logger.warning("StrategicOrch cycle failed: %s", exc, exc_info=True)

    def _detect_trigger(self, now: datetime) -> str:
        reports = self._load_session_reports()
        recent = sorted(reports, key=lambda r: r.get("date", ""), reverse=True)
        if len(recent) >= self._emergency_threshold:
            if all(r.get("overall_grade", "A") in ("D", "F")
                   for r in recent[:self._emergency_threshold]):
                return f"emergency:{self._emergency_threshold}_consecutive_D/F"
        return "weekly_scheduled"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> dict | None:
        if not text:
            return None
        try:
            t = text.strip()
            if t.startswith("```"):
                t = t.split("```", 2)[1]
                if t.startswith("json"):
                    t = t[4:]
                t = t.rsplit("```", 1)[0]
            return json.loads(t.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("StrategicOrch: failed to parse response: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def _write_baseline(self, parsed: dict, current_weights: dict[str, float]) -> None:
        """Write strategic_baseline.json consumed by TacticalMetaOrchestrator."""
        raw_weights = dict(parsed.get("new_baseline_weights") or {})
        if not raw_weights:
            return  # no weight changes proposed

        # Fill missing strategies from current
        for name in _STRATEGY_DESCRIPTIONS:
            if name not in raw_weights:
                raw_weights[name] = current_weights.get(name, 1.0 / len(_STRATEGY_DESCRIPTIONS))

        # Clamp movements to ±20% of current
        for name in list(raw_weights):
            cur = current_weights.get(name, raw_weights[name])
            if cur > 0:
                lo = cur * 0.80
                hi = cur * 1.20
                raw_weights[name] = max(lo, min(hi, raw_weights[name]))

        weights = _normalise_weights(raw_weights)

        corridor_pct = int(self._bounds.get("tactical_corridor_pct", 30))
        raw_corridors = dict(parsed.get("weight_corridors") or {})
        corridors: dict[str, int] = {}
        for name in _STRATEGY_DESCRIPTIONS:
            c = int(raw_corridors.get(name, corridor_pct))
            corridors[name] = max(10, min(50, c))

        baseline = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "week": parsed.get("week", ""),
            "trigger": parsed.get("trigger", ""),
            "strategy_weights": weights,
            "weight_corridors": corridors,
            "confidence": float(parsed.get("confidence", 0.5)),
            "regime_forecast": str(parsed.get("regime_forecast", "")),
            "strategic_notes": str(parsed.get("weight_rationale", "")),
        }
        with self._lock:
            self._baseline = baseline
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            path = os.path.join(self._output_dir, "strategic_baseline.json")
            with open(path, "w") as f:
                json.dump(baseline, f, indent=2)
            logger.info("StrategicOrch: baseline written to %s", path)
        except Exception as exc:
            logger.warning("StrategicOrch: could not write baseline: %s", exc)

        if _PROMETHEUS_OK:
            for strategy, w in weights.items():
                _STRATEGIC_WEIGHT.labels(strategy=strategy).set(w)

    def _load_baseline(self) -> None:
        """Load existing baseline from disk on startup."""
        try:
            path = os.path.join(self._output_dir, "strategic_baseline.json")
            if os.path.exists(path):
                with open(path) as f:
                    self._baseline = json.load(f)
                logger.info("StrategicOrch: loaded existing baseline from %s", path)
        except Exception as exc:
            logger.debug("StrategicOrch: could not load baseline: %s", exc)

    def get_baseline(self) -> dict:
        """Return the current strategic baseline (for TacticalMetaOrchestrator)."""
        with self._lock:
            return dict(self._baseline)

    # ------------------------------------------------------------------
    # Change queuing and application
    # ------------------------------------------------------------------

    def _queue_changes(self, parsed: dict, now: datetime) -> None:
        for ch in (parsed.get("parameter_changes") or []):
            param   = str(ch.get("parameter", ""))
            new_val = ch.get("proposed_value")
            rationale = str(ch.get("rationale", ""))
            if not param or new_val is None:
                continue
            if not self._within_bounds(param, new_val):
                logger.info("StrategicOrch: skipping %s=%s — out of bounds", param, new_val)
                continue
            apply_at = now + timedelta(seconds=self._apply_delay_sec)
            with self._lock:
                self._pending = [p for p in self._pending if p["param"] != param]
                self._pending.append({
                    "param": param, "new_val": new_val,
                    "rationale": rationale, "apply_at": apply_at,
                    "queued_at": now.isoformat(),
                })
            logger.info("StrategicOrch queued %s → %s in %.0fh | %s",
                        param, new_val, self._apply_delay_sec / 3600, rationale)

    def _apply_pending(self, now: datetime) -> None:
        with self._lock:
            ready = [p for p in self._pending if p["apply_at"] <= now]
            self._pending = [p for p in self._pending if p["apply_at"] > now]
        for change in ready:
            self._apply_change(change)

    def _apply_change(self, change: dict) -> None:
        param = change["param"]
        new_val = change["new_val"]
        keys = param.split(".")
        old_val = _get_nested(self._cfg, keys)
        if _set_nested(self._cfg, keys, new_val):
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "parameter": param, "old_value": old_val,
                "new_value": new_val, "rationale": change["rationale"],
                "source": "strategic",
            }
            with self._lock:
                self._applied_log.append(entry)
                if len(self._applied_log) > 50:
                    self._applied_log = self._applied_log[-50:]
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                log_path = os.path.join(self._output_dir, "changes.jsonl")
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
            logger.info("StrategicOrch applied %s: %s → %s | %s",
                        param, old_val, new_val, change["rationale"])
        else:
            logger.warning("StrategicOrch: could not apply %s — path not found", param)

    def _within_bounds(self, param: str, value: Any) -> bool:
        bounds = self._bounds.get(param)
        if bounds is None:
            return False
        try:
            return float(bounds[0]) <= float(value) <= float(bounds[1])
        except (TypeError, ValueError, IndexError):
            return False

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_session_reports(self) -> list[dict]:
        try:
            p = Path(self._session_dir)
            if not p.exists():
                return []
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            reports = []
            for f in sorted(p.glob("report_*.json"), reverse=True):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    date_str = data.get("date", "")
                    if date_str:
                        try:
                            report_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                            if report_date < cutoff:
                                continue
                        except ValueError:
                            pass
                    reports.append(data)
                    if len(reports) >= 7:
                        break
                except Exception:
                    continue
            return reports
        except Exception as exc:
            logger.debug("StrategicOrch: could not load session reports: %s", exc)
            return []

    def _load_tactical_changes(self) -> list[dict]:
        try:
            p = Path(self._tactical_changes_log)
            if not p.exists():
                return []
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            changes = []
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ch = json.loads(line)
                    ts_str = ch.get("ts", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            if ts < cutoff:
                                continue
                        except ValueError:
                            pass
                    changes.append(ch)
                except (json.JSONDecodeError, ValueError):
                    continue
            return changes
        except Exception as exc:
            logger.debug("StrategicOrch: could not load tactical changes: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Persistence and API
    # ------------------------------------------------------------------

    def _save_result(self, parsed: dict) -> None:
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            path = os.path.join(self._output_dir, "strategic_last_result.json")
            with open(path, "w") as f:
                json.dump(parsed, f, indent=2)
        except Exception as exc:
            logger.debug("StrategicOrch: could not save result: %s", exc)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "last_run": self._last_run.isoformat() if self._last_run else None,
                "last_result": dict(self._last_result),
                "baseline": dict(self._baseline),
                "pending_changes": [
                    {k: v for k, v in p.items() if k != "apply_at"}
                    | {"apply_in_seconds": max(0, int((p["apply_at"]
                       - datetime.now(timezone.utc)).total_seconds()))}
                    for p in self._pending
                ],
                "applied_log": list(self._applied_log[-20:]),
            }


# Backward-compatible alias for any code that imports the old class name
MetaOrchestrator = StrategicOrchestrator

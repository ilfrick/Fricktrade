# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Tactical Meta Orchestrator — 15-minute config tuner and tactical strategist.

Runs every 15 minutes. Reads live performance metrics, macro regime cache,
and the last post-session report, then fires one Gemini call to propose
targeted config adjustments that improve short-term performance.

Two-tier change application:
  - Operational params (cooldowns, margins, timeouts) → applied immediately
  - Strategy weights, stop/TP, hold params → queued, applied after apply_delay_minutes

All changes are bounded by explicit safe ranges in config. Changes are logged
to /data/reports/meta_orch/changes.jsonl for audit and Grafana display.

The weekly MetaOrchestrator (#4) remains the long-horizon strategic layer;
this orchestrator operates within ±30% corridors around those baselines
(or self-imposed ±30% when no weekly baseline exists).
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
    _HEALTH_GRADE    = Gauge("meta_orch_health_grade", "Health grade: 0=green 1=yellow 2=red")
    _PENDING_COUNT   = Gauge("meta_orch_pending_changes_count", "Changes queued, awaiting delay")
    _CYCLE_LATENCY   = Gauge("meta_orch_last_cycle_latency_seconds", "Last LLM call duration")
    _CHANGES_APPLIED = Counter("meta_orch_changes_applied_total", "Config changes applied", ["parameter"])
    _WEIGHT_GAUGE    = Gauge("meta_orch_strategy_weight", "Live strategy weight", ["strategy"])
    _OVERRIDE_WIN    = Gauge("meta_orch_llm_override_win_rate", "LLM override win rate (0-1)")
    _COMBINE_WIN     = Gauge("meta_orch_combine_win_rate", "Combine-signals win rate (0-1)")
    _PROMETHEUS_OK   = True
except Exception:
    _PROMETHEUS_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the TACTICAL STRATEGIST for an autonomous crypto and equity trading system.
You run every 15 minutes and make short-horizon adjustments to configuration
parameters to improve performance over the next 1-2 hours.

YOUR DECISION FRAMEWORK (apply in priority order):
1. REGIME FIT — Are the active strategies right for the current macro regime?
   In a trending regime (Hurst > 0.60) favour momentum strategies.
   In range-bound (Hurst < 0.45) favour mean-reversion and tighter exits.
   In crisis (VIX > 30) reduce size, tighten stops, extend min_hold.
2. LIVE PERFORMANCE — Boost what is winning right now. Suppress what is losing.
   Require at least 5 fills before adjusting any strategy's weight.
   If no strategy has ≥5 fills, skip weight changes and say so.
3. EXIT QUALITY — Are stops being hit too early? Is take-profit leaving money on
   the table? Adjust stop/TP/hold to match the current ATR environment.
4. OPERATIONAL WASTE — Fix retry loops and stuck orders only when they are also
   costing money. Operational fixes have lower priority than performance tuning.

HARD CONSTRAINTS:
- Strategy weights must sum to 1.0 after adjustment (renormalise if needed).
- No weight below 0.03 or above 0.45.
- No parameter may move more than 30% from its CURRENT value in one call.
- If combine_mode is "vote", strategy weight changes have no immediate effect —
  state this and skip weight changes unless you have a strong reason.
- Never change: broker credentials, API keys, enabled/disabled brokers, or any
  parameter not listed in the safe bounds table provided.
- If the data does not support any change, say so — do not tune for the sake of it.

Respond ONLY with valid JSON — no markdown, no text outside the JSON object.\
"""


def _build_user_prompt(metrics: dict, bounds: dict, combine_mode: str) -> str:
    ts = metrics.get("timestamp", "unknown")
    window = metrics.get("window_minutes", 15)

    # Regime block
    regime = metrics.get("regime") or {}
    regime_name = regime.get("name", "unknown")
    regime_conf = regime.get("confidence", 0.0)
    regime_rat  = regime.get("rationale", "n/a")
    regime_age  = regime.get("age_minutes", "?")
    regime_mult = json.dumps(regime.get("weight_overrides") or {})
    vix   = regime.get("vix", "?")
    dgs10 = regime.get("dgs10", "?")
    dxy   = regime.get("dxy", "?")

    # Session context
    session_grade   = metrics.get("last_session_grade", "n/a")
    session_finding = metrics.get("last_session_finding", "n/a")
    strategic_notes = metrics.get("strategic_notes", "none")

    # Strategy perf table
    strat_rows = []
    for name, p in (metrics.get("strategy_perf") or {}).items():
        w = (metrics.get("current_weights") or {}).get(name, 0.0)
        strat_rows.append(
            f"  {name:<28} n={p.get('n',0):>3}  win={p.get('win_rate',0)*100:.0f}%"
            f"  avg_pnl={p.get('avg_pnl_pct',0):+.2f}%"
            f"  hold={p.get('avg_hold_min',0):.0f}min  weight={w:.3f}"
        )
    strat_table = "\n".join(strat_rows) if strat_rows else "  (no fills yet)"

    # Portfolio
    equity        = metrics.get("equity", 0)
    crypto_pct    = metrics.get("crypto_exposure_pct", 0)
    crypto_cap    = metrics.get("crypto_cap_pct", 50)
    pos_count     = metrics.get("positions_count", 0)
    pos_summary   = metrics.get("positions_summary", "none")
    session_pnl   = metrics.get("session_pnl_pct", 0.0)
    llm_ovr_rate  = metrics.get("llm_override_rate", 0.0) * 100
    llm_ovr_win   = metrics.get("llm_override_win_rate", 0.0) * 100
    combine_win   = metrics.get("combine_win_rate", 0.0) * 100

    # Per-broker breakdown
    broker_rows = []
    for bname, bd in sorted((metrics.get("brokers") or {}).items()):
        b_eq   = bd.get("estimated_equity", 0)
        b_cash = bd.get("available_cash", 0)
        b_cpct = bd.get("crypto_exposure_pct", 0)
        b_cap  = bd.get("max_crypto_pct", 40)
        b_dd   = bd.get("current_drawdown_pct", 0)
        b_n    = bd.get("open_positions", 0)
        b_dis  = ", ".join(bd.get("disabled_strategies") or []) or "none"
        broker_rows.append(
            f"  {bname:<22} eq≈${b_eq:,.0f}  cash=${b_cash:,.0f}"
            f"  crypto={b_cpct:.1f}%/{b_cap:.0f}%  dd={b_dd:.1f}%  pos={b_n}"
            + (f"  disabled=[{b_dis}]" if b_dis != "none" else "")
        )
    brokers_str = "\n".join(broker_rows) if broker_rows else "  (no broker data)"

    # Indicators
    med_rsi   = metrics.get("median_rsi", 50)
    med_atr   = metrics.get("median_atr_pct", 1.0)
    med_hurst = metrics.get("median_hurst", 0.5)

    # Order flow
    attempted    = metrics.get("orders_attempted", 0)
    fill_rate    = metrics.get("fill_rate_pct", 0)
    rejected     = metrics.get("orders_rejected", 0)
    ins_stable   = metrics.get("insuff_stablecoin", 0)
    ins_cash     = metrics.get("insuff_cash", 0)
    pos_lim      = metrics.get("pos_limit_blocks", 0)
    lev_cap      = metrics.get("leverage_cap_blocks", 0)
    timedout     = metrics.get("orders_timedout", 0)
    stuck_cool   = metrics.get("stuck_cooldowns", 0)
    exit_bkoff   = metrics.get("exit_backoffs", 0)
    dust_cnt     = metrics.get("dust_count", 0)

    # Bounds table
    bounds_lines = []
    for k, v in sorted(bounds.items()):
        if k.startswith("strategy_weight"):
            continue
        cur = _get_nested(metrics.get("current_config", {}), k.split("."))
        bounds_lines.append(f"  {k:<52} [{v[0]}, {v[1]}]  current: {cur}")
    bounds_str = "\n".join(bounds_lines)

    current_weights_json = json.dumps(metrics.get("current_weights") or {}, indent=4)
    sw_min = bounds.get("strategy_weight_min", 0.03)
    sw_max = bounds.get("strategy_weight_max", 0.45)

    # Strategic baseline corridors (if available)
    strategic = metrics.get("strategic_baseline") or {}
    baseline_weights = strategic.get("strategy_weights") or {}
    corridors = strategic.get("weight_corridors") or {}
    regime_forecast = strategic.get("regime_forecast", "")
    strategic_notes = strategic.get("strategic_notes", "")
    if baseline_weights:
        corridor_lines = []
        for name, bw in sorted(baseline_weights.items()):
            cpct = corridors.get(name, 30)
            lo = round(bw * (1 - cpct / 100), 4)
            hi = round(bw * (1 + cpct / 100), 4)
            corridor_lines.append(f"  {name:<28} baseline={bw:.3f}  allowed=[{lo:.3f},{hi:.3f}]")
        strategic_block = (
            f"\n── STRATEGIC BASELINES (from weekly review) ───────────────\n"
            + "\n".join(corridor_lines)
            + f"\nRegime forecast: {regime_forecast}\n"
            + (f"Strategic notes: {strategic_notes[:120]}\n" if strategic_notes else "")
            + "Weight changes MUST stay within the allowed corridors above.\n"
        )
    else:
        strategic_block = (
            "\n── STRATEGIC BASELINES ─────────────────────────────────────\n"
            "No weekly baseline yet. Self-impose ±30% from current weights.\n"
        )

    vote_note = ""
    if combine_mode == "vote":
        vote_note = (
            "\nNOTE: combine_mode is currently 'vote' — strategy weights do NOT affect "
            "signal combination. Weight changes will be stored but have no immediate effect. "
            "Skip weight changes unless you intend to recommend switching combine_mode.\n"
        )

    return f"""\
TACTICAL REVIEW — {ts} UTC  |  Window: last {window} min
{'━'*65}
{vote_note}{strategic_block}
── MACRO CONTEXT (regime classifier, age: {regime_age}min) ─────
Regime:          {regime_name}  (confidence: {regime_conf:.0%})
Rationale:       {regime_rat}
Weight overrides:{regime_mult}
VIX: {vix}  |  10Y: {dgs10}%  |  DXY: {dxy}

── STRATEGIC CONTEXT ───────────────────────────────────────────
Last session grade:  {session_grade}
Top session finding: {session_finding}
Strategic notes:     {strategic_notes}

── LIVE STRATEGY PERFORMANCE ───────────────────────────────────
  strategy                       n    win   avg_pnl   hold    weight
{strat_table}

── PORTFOLIO STATE ─────────────────────────────────────────────
Equity:           ${equity:,.0f}
Crypto exposure:  {crypto_pct:.1f}% / {crypto_cap:.0f}% cap
Open positions:   {pos_count}  ({pos_summary})
Session P&L:      {session_pnl:+.2f}%
LLM override rate:{llm_ovr_rate:.0f}%  win={llm_ovr_win:.0f}%  (baseline combine={combine_win:.0f}%)

── BROKERS ─────────────────────────────────────────────────────
{brokers_str}

── CURRENT INDICATORS (median across held positions) ───────────
Median RSI:   {med_rsi:.0f}
Median ATR%:  {med_atr:.2f}%
Median Hurst: {med_hurst:.2f}

── OPERATIONAL HEALTH ──────────────────────────────────────────
Orders attempted:       {attempted}
Fill rate:              {fill_rate:.0f}%
Rejected:               {rejected}  (insuff_stable={ins_stable} insuff_cash={ins_cash} pos_limit={pos_lim} leverage_cap={lev_cap})
Timed out:              {timedout}  (stuck cooldowns: {stuck_cool})
Exit backoffs:          {exit_bkoff}
Dust positions:         {dust_cnt}

── SAFE PARAMETER BOUNDS ───────────────────────────────────────
Strategy weights (must sum to 1.0, each in [{sw_min}, {sw_max}], max ±30% move):
{current_weights_json}

Config parameters (param → [min, max], current value):
{bounds_str}

── RESPONSE FORMAT ─────────────────────────────────────────────
{{
  "timestamp": "{ts}",
  "health_grade": "<green|yellow|red>",
  "regime_alignment": "<well_aligned|partially_aligned|misaligned>",
  "performance_summary": "<2 sentences on what is working and what is not>",
  "primary_finding": "<single most important thing to fix or exploit>",
  "proposed_weight_changes": {{
    "<strategy_name>": <new_weight_float>
    // only include strategies whose weight changes; omit unchanged ones
    // all weights including unchanged ones must sum to 1.0
  }},
  "proposed_config_changes": [
    {{
      "parameter": "<exact dotted path>",
      "current_value": <current>,
      "proposed_value": <new>,
      "rationale": "<one sentence citing specific metric>"
    }}
  ],
  "anomalies": ["<anything needing human attention>"],
  "no_change_reason": "<if both lists empty, explain why>"
}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMMEDIATE_PARAMS = {
    "execution.stuck_cooldown_minutes",
    "execution.insufficient_stablecoin_cooldown_minutes",
    "trading_limits.crypto_order_margin",
    "data.timeout_seconds",
}


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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TacticalMetaOrchestrator:
    """15-minute tactical config tuner and performance strategist."""

    def __init__(self, llm_client: Any, cfg: dict) -> None:
        self._llm  = llm_client
        self._cfg  = cfg  # live reference — mutations take effect immediately
        tmo_cfg = cfg.get("tactical_meta_orchestrator", {}) or {}
        self._enabled          = bool(tmo_cfg.get("enabled", True))
        self._interval_sec     = int(tmo_cfg.get("interval_minutes", 15)) * 60
        self._apply_delay_sec  = int(tmo_cfg.get("apply_delay_minutes", 5)) * 60
        self._backend          = str(tmo_cfg.get("backend", "gemini"))
        self._model            = tmo_cfg.get("model") or None
        self._session_dir      = str(tmo_cfg.get("session_reports_dir",
                                    "/data/reports/session"))
        self._output_dir       = str(tmo_cfg.get("output_dir",
                                    "/data/reports/meta_orch"))
        self._bounds: dict     = dict(tmo_cfg.get("bounds", {}))
        # Runtime state
        self._last_run: datetime | None = None
        self._pending: list[dict]       = []   # {param, old, new, rationale, apply_at, type}
        self._applied_log: list[dict]   = []   # last 50 applied changes (for API)
        self._last_result: dict         = {}
        self._lock = threading.RLock()  # RLock: _run_cycle holds lock then calls _enqueue_change (re-entrant)
        # Reference to trader's live _config_strategy_weights dict (set externally)
        self._weights_ref: dict[str, float] | None = None
        # Strategic baseline from StrategicOrchestrator (for corridor enforcement)
        self._last_strategic_baseline: dict = {}
        # Order flow counters — reset by trader each call
        self._order_flow_window: dict[str, int] = {}
        logger.info("TacticalMetaOrchestrator initialised (interval=%smin, delay=%smin)",
                    self._interval_sec // 60, self._apply_delay_sec // 60)

    def set_weights_ref(self, weights: dict[str, float]) -> None:
        """Called by trader to share its live _config_strategy_weights dict."""
        self._weights_ref = weights

    # ------------------------------------------------------------------
    # Main entry point (called from trader main loop)
    # ------------------------------------------------------------------

    def maybe_run(self, metrics: dict) -> None:
        """Run a tactical review cycle if the interval has elapsed."""
        if not self._enabled:
            return
        now = datetime.now(timezone.utc)
        # First apply any pending changes whose delay has passed
        self._apply_pending(now)
        if self._last_run and (now - self._last_run).total_seconds() < self._interval_sec:
            return
        self._last_run = now
        # Run in a background thread so the trading loop is never blocked
        t = threading.Thread(target=self._run_cycle, args=(metrics, now), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Internal cycle
    # ------------------------------------------------------------------

    def _run_cycle(self, metrics: dict, cycle_start: datetime) -> None:
        try:
            metrics = dict(metrics)
            metrics["timestamp"] = cycle_start.strftime("%Y-%m-%d %H:%M")
            metrics["window_minutes"] = self._interval_sec // 60
            # Cache strategic baseline for corridor enforcement in _queue_changes
            self._last_strategic_baseline = metrics.get("strategic_baseline") or {}
            # Inject last post-session report
            metrics.update(self._load_session_context())
            # Inject current config snapshot for bounds display
            metrics["current_config"] = self._cfg
            combine_mode = str(metrics.get("combine_mode", "weighted"))

            prompt = _build_user_prompt(metrics, self._bounds, combine_mode)
            t0 = time.monotonic()
            response = self._llm.complete(
                backend=self._backend,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=self._model,
                max_tokens=1024,
            )
            latency = time.monotonic() - t0
            if _PROMETHEUS_OK:
                _CYCLE_LATENCY.set(latency)

            parsed = self._parse_response(response.content)
            if parsed:
                with self._lock:
                    self._last_result = parsed
                    self._queue_changes(parsed, datetime.now(timezone.utc))
                self._save_result(parsed)
                self._update_prometheus(parsed, metrics)
                logger.info(
                    "TacticalMetaOrch: grade=%s  finding=%s  changes=%d pending=%d",
                    parsed.get("health_grade", "?"),
                    parsed.get("primary_finding", "")[:80],
                    len(parsed.get("proposed_config_changes") or [])
                    + len(parsed.get("proposed_weight_changes") or {}),
                    len(self._pending),
                )
        except Exception as exc:
            logger.warning("TacticalMetaOrch cycle failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> dict | None:
        if not text:
            return None
        try:
            # Strip markdown fences if present
            t = text.strip()
            if t.startswith("```"):
                t = t.split("```", 2)[1]
                if t.startswith("json"):
                    t = t[4:]
                t = t.rsplit("```", 1)[0]
            return json.loads(t.strip())
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("TacticalMetaOrch: failed to parse response: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Change queuing and application
    # ------------------------------------------------------------------

    def _queue_changes(self, parsed: dict, now: datetime) -> None:
        weight_changes = parsed.get("proposed_weight_changes") or {}
        config_changes = parsed.get("proposed_config_changes") or []

        # Build strategic corridors if available
        strategic = getattr(self, "_last_strategic_baseline", {}) or {}
        baseline_weights = strategic.get("strategy_weights") or {}
        corridors = strategic.get("weight_corridors") or {}

        for param, new_val in weight_changes.items():
            new_val = float(new_val)
            # Enforce strategic corridor if baseline exists
            if param in baseline_weights:
                bw = float(baseline_weights[param])
                cpct = float(corridors.get(param, 30)) / 100.0
                lo = max(0.03, bw * (1 - cpct))
                hi = min(0.45, bw * (1 + cpct))
                if not (lo <= new_val <= hi):
                    new_val = max(lo, min(hi, new_val))
                    logger.debug("TacticalMetaOrch: clipped %s to strategic corridor [%.3f, %.3f] → %.3f",
                                 param, lo, hi, new_val)
            self._enqueue_change(
                param=f"orchestrator.strategy_weights.{param}",
                new_val=new_val,
                rationale="weight adjustment from tactical review",
                now=now,
                immediate=False,
            )

        for ch in config_changes:
            param = str(ch.get("parameter", ""))
            new_val = ch.get("proposed_value")
            rationale = str(ch.get("rationale", ""))
            if not param or new_val is None:
                continue
            if not self._within_bounds(param, new_val):
                logger.info("TacticalMetaOrch: skipping %s=%s — out of bounds", param, new_val)
                continue
            self._enqueue_change(
                param=param,
                new_val=new_val,
                rationale=rationale,
                now=now,
                immediate=(param in _IMMEDIATE_PARAMS),
            )

    def _enqueue_change(self, param: str, new_val: Any, rationale: str,
                        now: datetime, immediate: bool) -> None:
        apply_at = now if immediate else now + timedelta(seconds=self._apply_delay_sec)
        with self._lock:
            # Deduplicate: replace any existing pending entry for same param
            self._pending = [p for p in self._pending if p["param"] != param]
            self._pending.append({
                "param": param,
                "new_val": new_val,
                "rationale": rationale,
                "apply_at": apply_at,
                "immediate": immediate,
                "queued_at": now.isoformat(),
            })
        if _PROMETHEUS_OK:
            _PENDING_COUNT.set(len(self._pending))
        logger.debug("TacticalMetaOrch queued %s → %s (%s)",
                     param, new_val, "immediate" if immediate else f"in {self._apply_delay_sec//60}min")

    def _apply_pending(self, now: datetime) -> None:
        with self._lock:
            ready = [p for p in self._pending if p["apply_at"] <= now]
            self._pending = [p for p in self._pending if p["apply_at"] > now]
        if _PROMETHEUS_OK:
            _PENDING_COUNT.set(len(self._pending))
        for change in ready:
            self._apply_change(change)

    def _apply_change(self, change: dict) -> None:
        param = change["param"]
        new_val = change["new_val"]
        keys = param.split(".")

        # Special handling for strategy weights
        if param.startswith("orchestrator.strategy_weights."):
            strategy = keys[-1]
            if self._weights_ref is not None:
                old_val = self._weights_ref.get(strategy)
                self._weights_ref[strategy] = float(new_val)
                # Also persist to cfg so the web UI shows the update
                sw = self._cfg.setdefault("orchestrator", {}).setdefault("strategy_weights", {})
                sw[strategy] = float(new_val)
                self._log_applied(param, old_val, new_val, change["rationale"])
                if _PROMETHEUS_OK:
                    _WEIGHT_GAUGE.labels(strategy=strategy).set(float(new_val))
                    _CHANGES_APPLIED.labels(parameter=param).inc()
                logger.info("TacticalMetaOrch applied weight %s: %s → %s | %s",
                            strategy, old_val, new_val, change["rationale"])
            return

        # General config path
        old_val = _get_nested(self._cfg, keys)
        applied = _set_nested(self._cfg, keys, new_val)
        if applied:
            self._log_applied(param, old_val, new_val, change["rationale"])
            if _PROMETHEUS_OK:
                _CHANGES_APPLIED.labels(parameter=param).inc()
            logger.info("TacticalMetaOrch applied %s: %s → %s | %s",
                        param, old_val, new_val, change["rationale"])
        else:
            logger.warning("TacticalMetaOrch: failed to apply %s — path not found in cfg", param)

    # ------------------------------------------------------------------
    # Bounds validation
    # ------------------------------------------------------------------

    def _within_bounds(self, param: str, value: Any) -> bool:
        bounds = self._bounds.get(param)
        if bounds is None:
            # Unknown param — reject to be safe
            return False
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
            return lo <= float(value) <= hi
        except (TypeError, ValueError, IndexError):
            return False

    # ------------------------------------------------------------------
    # Post-session context loader
    # ------------------------------------------------------------------

    def _load_session_context(self) -> dict:
        try:
            p = Path(self._session_dir)
            if not p.exists():
                return {}
            reports = sorted(p.glob("report_*.json"), reverse=True)
            if not reports:
                return {}
            with open(reports[0]) as f:
                data = json.load(f)
            grade = data.get("overall_grade", "n/a")
            findings = data.get("key_findings") or []
            top = findings[0].get("finding", "n/a") if findings else "n/a"
            notes = "; ".join(
                f"{a.get('strategy','?')}: {a.get('should_adjust_weight','?')}"
                for a in (data.get("strategy_assessments") or [])[:4]
            )
            return {
                "last_session_grade": grade,
                "last_session_finding": top,
                "strategic_notes": notes or "n/a",
            }
        except Exception as exc:
            logger.debug("TacticalMetaOrch: could not load session report: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Prometheus and logging
    # ------------------------------------------------------------------

    def _update_prometheus(self, parsed: dict, metrics: dict) -> None:
        if not _PROMETHEUS_OK:
            return
        grade_map = {"green": 0, "yellow": 1, "red": 2}
        _HEALTH_GRADE.set(grade_map.get(parsed.get("health_grade", "green"), 0))
        _OVERRIDE_WIN.set(float(metrics.get("llm_override_win_rate", 0)))
        _COMBINE_WIN.set(float(metrics.get("combine_win_rate", 0)))
        for strategy, w in (metrics.get("current_weights") or {}).items():
            _WEIGHT_GAUGE.labels(strategy=strategy).set(float(w))

    def _log_applied(self, param: str, old_val: Any, new_val: Any, rationale: str) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "parameter": param,
            "old_value": old_val,
            "new_value": new_val,
            "rationale": rationale,
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
        except Exception as exc:
            logger.debug("TacticalMetaOrch: could not write change log: %s", exc)

    def _save_result(self, parsed: dict) -> None:
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            path = os.path.join(self._output_dir, "last_result.json")
            with open(path, "w") as f:
                json.dump(parsed, f, indent=2)
        except Exception as exc:
            logger.debug("TacticalMetaOrch: could not save result: %s", exc)

    # ------------------------------------------------------------------
    # API / status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current state for the /api/meta_orch endpoint."""
        with self._lock:
            return {
                "last_run": self._last_run.isoformat() if self._last_run else None,
                "last_result": dict(self._last_result),
                "pending_changes": [
                    {k: v for k, v in p.items() if k != "apply_at"}
                    | {"apply_in_seconds": max(0, int((p["apply_at"]
                       - datetime.now(timezone.utc)).total_seconds()))}
                    for p in self._pending
                ],
                "applied_log": list(self._applied_log[-20:]),
            }

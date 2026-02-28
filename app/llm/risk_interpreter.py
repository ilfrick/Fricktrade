# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
LLM-powered risk event interpretation.

When the drift monitor or risk manager detects an anomaly, this module
provides qualitative analysis to decide whether it's a real structural
break or transient noise.

Uses Claude with low temperature (0.1) for maximum determinism.
Bypasses the daily budget circuit breaker (critical=True) since risk
events require a response regardless of cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


RISK_SYSTEM_PROMPT = """\
You are a risk analyst for an automated trading system. When the system's
drift monitor or risk manager triggers an alert, you analyze the context
to determine:

1. Is this a genuine structural break or transient noise?
2. What is the likely cause?
3. Should the system pause, reduce exposure, or continue?
4. Is the auto-rollback to best model appropriate, or is the market
   fundamentally different now?

Be decisive. Trading systems need clear recommendations, not hedging.
Respond ONLY with valid JSON.
"""

RISK_USER_TEMPLATE = """\
RISK ALERT TRIGGERED
━━━━━━━━━━━━━━━━━━━
Alert type: {alert_type}
Severity: {severity}
Timestamp: {timestamp}
Description: {description}

DRIFT METRICS:
  Feature drift z-score: {feature_drift_z:.2f}
  PnL drift z-score: {pnl_drift_z:.2f}
  Rolling Sharpe (20d): {rolling_sharpe:.2f}
  Consecutive losses: {consecutive_losses}

CURRENT REGIME:
  HMM state: {hmm_state}
  VIX: {vix:.1f} (change: {vix_change:+.1f})
  SPY: {spy_change:+.2f}%
  Sector rotation: {sector_rotation}

RECENT PERFORMANCE (last 5 sessions):
{recent_perf}

CURRENT POSITIONS:
{positions}

SYSTEM STATE:
  Active model version: {model_version}
  Best model version: {best_model_version}
  Auto-rollback available: {rollback_available}

Respond with:
{{
  "diagnosis": "<structural_break|transient_noise|regime_shift|data_issue>",
  "confidence": <float 0.0 to 1.0>,
  "explanation": "<paragraph explaining the diagnosis>",
  "likely_cause": "<description>",
  "recommended_action": "<continue|reduce_exposure|pause_trading|rollback_model|emergency_flatten>",
  "position_action": "<hold|reduce_50pct|close_losing|flatten_all|no_change>",
  "should_rollback_model": <true|false>,
  "rollback_rationale": "<why or why not>",
  "resume_conditions": "<what conditions should be met to resume normal ops>",
  "urgency": "<immediate|within_hour|end_of_session>"
}}
"""


@dataclass
class RiskInterpretation:
    diagnosis: str
    confidence: float
    explanation: str
    likely_cause: str
    recommended_action: str
    position_action: str
    should_rollback_model: bool
    rollback_rationale: str
    resume_conditions: str
    urgency: str
    cost_usd: float


class RiskEventInterpreter:
    """
    Interprets risk events with qualitative LLM reasoning.

    Integrates with the existing drift monitor and risk manager to
    provide intelligent triage instead of simple auto-rollback.

    Note: calls are marked critical=True to bypass the daily budget
    circuit breaker — risk responses must not be suppressed by cost.
    """

    def __init__(self, llm_client: LLMClient, backend: str = "claude"):
        self.llm = llm_client
        self.backend = backend

    def interpret(
        self,
        alert_type: str,
        severity: str,
        timestamp: str,
        description: str,
        feature_drift_z: float = 0.0,
        pnl_drift_z: float = 0.0,
        rolling_sharpe: float = 0.0,
        consecutive_losses: int = 0,
        hmm_state: str = "unknown",
        vix: float = 20.0,
        vix_change: float = 0.0,
        spy_change: float = 0.0,
        sector_rotation: str = "none",
        recent_performance: Optional[list[dict]] = None,
        positions: Optional[list[dict]] = None,
        model_version: str = "current",
        best_model_version: str = "current",
        rollback_available: bool = False,
    ) -> Optional[RiskInterpretation]:
        """Interpret a risk alert with full context."""
        recent_str = "\n".join(
            f"  {p['date']}: PnL ${p.get('pnl', 0):+,.2f} | "
            f"Trades: {p.get('trades', 0)} | WR: {p.get('win_rate', 0):.0f}%"
            for p in (recent_performance or [])
        ) or "  No recent performance data"

        positions_str = "\n".join(
            f"  {p['symbol']:6s} {p.get('side', 'long'):5s} qty={p.get('quantity', 0)} "
            f"entry=${p.get('entry_price', 0):.2f} "
            f"current=${p.get('current_price', 0):.2f} "
            f"unrealized=${p.get('unrealized_pnl', 0):+,.2f}"
            for p in (positions or [])
        ) or "  No open positions"

        user_prompt = RISK_USER_TEMPLATE.format(
            alert_type=alert_type,
            severity=severity,
            timestamp=timestamp,
            description=description,
            feature_drift_z=feature_drift_z,
            pnl_drift_z=pnl_drift_z,
            rolling_sharpe=rolling_sharpe,
            consecutive_losses=consecutive_losses,
            hmm_state=hmm_state,
            vix=vix,
            vix_change=vix_change,
            spy_change=spy_change,
            sector_rotation=sector_rotation,
            recent_perf=recent_str,
            positions=positions_str,
            model_version=model_version,
            best_model_version=best_model_version,
            rollback_available=rollback_available,
        )

        try:
            response = self.llm.complete(
                backend=self.backend,
                system_prompt=RISK_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=2048,
                temperature=0.1,
                critical=True,  # Bypass daily budget for risk calls
            )
            data = response.parse_json()

            return RiskInterpretation(
                diagnosis=data["diagnosis"],
                confidence=float(data["confidence"]),
                explanation=data["explanation"],
                likely_cause=data["likely_cause"],
                recommended_action=data["recommended_action"],
                position_action=data["position_action"],
                should_rollback_model=bool(data["should_rollback_model"]),
                rollback_rationale=data["rollback_rationale"],
                resume_conditions=data["resume_conditions"],
                urgency=data["urgency"],
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error("Risk interpretation failed: %s", e)
            return None

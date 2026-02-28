# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Post-session trade analysis using LLM.

Runs at end of each trading day to identify patterns, errors, and suggest
adjustments. Replaces manual daily review. Uses Gemini for its large context
window, which can handle full trade logs.

Output is saved as JSON to data/session_reports/ for archival and
meta-orchestrator consumption.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


ANALYST_SYSTEM_PROMPT = """\
You are a senior quantitative trader reviewing today's automated trading
session. You analyze trade logs with the precision of a prop desk risk
manager. Your goal is to identify:

1. Recurring error patterns (false signals, bad timing, wrong sizing)
2. Strategy-specific performance issues
3. Regime-strategy mismatches
4. Risk management effectiveness
5. Concrete, actionable improvements

Be brutally honest. Quantify everything. Do not sugarcoat losses.
Prioritize findings by estimated PnL impact.

Respond ONLY with valid JSON, no markdown, no commentary.
"""

ANALYST_USER_TEMPLATE = """\
SESSION SUMMARY — {date}
━━━━━━━━━━━━━━━━━━━━━━
Market regime: {regime}
VIX: {vix:.1f} | SPY: {spy_change:+.2f}%
Session equity: ${start_equity:,.2f} → ${end_equity:,.2f} ({pnl_pct:+.2f}%)
Total trades: {total_trades} | Win rate: {win_rate:.1f}%
Gross PnL: ${gross_pnl:+,.2f} | Fees: ${fees:,.2f} | Net: ${net_pnl:+,.2f}

STRATEGY BREAKDOWN:
{strategy_breakdown}

TRADE LOG (chronological):
{trade_log}

ORCHESTRATOR DECISIONS:
{orchestrator_log}

RISK EVENTS:
{risk_events}

Respond with:
{{
  "date": "{date}",
  "overall_grade": "<A/B/C/D/F>",
  "key_findings": [
    {{
      "finding": "<description>",
      "severity": "<critical|high|medium|low>",
      "estimated_pnl_impact": <float>,
      "affected_strategy": "<strategy name or 'system'>",
      "evidence": "<specific trade IDs or patterns>"
    }}
  ],
  "strategy_assessments": [
    {{
      "strategy": "<name>",
      "grade": "<A-F>",
      "trades": <int>,
      "pnl": <float>,
      "assessment": "<one paragraph>",
      "should_adjust_weight": "<increase|decrease|maintain>",
      "suggested_weight_change": <float between -0.3 and 0.3>
    }}
  ],
  "parameter_suggestions": [
    {{
      "parameter": "<config path>",
      "current_value": "<current>",
      "suggested_value": "<new>",
      "rationale": "<why>"
    }}
  ],
  "regime_analysis": "<how well did the system adapt to today's regime>",
  "tomorrow_recommendations": [
    "<actionable recommendation for tomorrow's session>"
  ]
}}
"""


@dataclass
class SessionAnalysis:
    date: str
    overall_grade: str
    key_findings: list[dict]
    strategy_assessments: list[dict]
    parameter_suggestions: list[dict]
    regime_analysis: str
    tomorrow_recommendations: list[str]
    cost_usd: float
    latency_ms: float


class PostSessionAnalyst:
    """
    Analyzes each trading session and produces actionable insights.

    Output is saved to disk and consumed by MetaOrchestrator for
    weekly weight rebalancing.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        backend: str = "gemini",
        output_dir: str = "data/session_reports",
    ):
        self.llm = llm_client
        self.backend = backend
        self.output_dir = output_dir

    def analyze(
        self,
        date: str,
        regime: str,
        vix: float,
        spy_change: float,
        start_equity: float,
        end_equity: float,
        trades: list[dict],
        strategy_stats: dict,
        orchestrator_decisions: list[dict],
        risk_events: list[dict],
        fees: float,
        max_trades_in_context: int = 100,
    ) -> Optional[SessionAnalysis]:
        """Run post-session analysis."""
        gross_pnl = end_equity - start_equity + fees
        net_pnl = end_equity - start_equity
        pnl_pct = (net_pnl / start_equity) * 100 if start_equity else 0
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        win_rate = (wins / len(trades) * 100) if trades else 0

        user_prompt = ANALYST_USER_TEMPLATE.format(
            date=date,
            regime=regime,
            vix=vix,
            spy_change=spy_change,
            start_equity=start_equity,
            end_equity=end_equity,
            pnl_pct=pnl_pct,
            total_trades=len(trades),
            win_rate=win_rate,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=net_pnl,
            strategy_breakdown=self._format_strategy_stats(strategy_stats),
            trade_log=self._format_trades(trades, max_trades_in_context),
            orchestrator_log=self._format_orchestrator(orchestrator_decisions),
            risk_events=self._format_risk_events(risk_events),
        )

        try:
            response = self.llm.complete(
                backend=self.backend,
                system_prompt=ANALYST_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=8192,
                temperature=0.2,
            )
            data = response.parse_json()

            result = SessionAnalysis(
                date=data["date"],
                overall_grade=data["overall_grade"],
                key_findings=data.get("key_findings", []),
                strategy_assessments=data.get("strategy_assessments", []),
                parameter_suggestions=data.get("parameter_suggestions", []),
                regime_analysis=data.get("regime_analysis", ""),
                tomorrow_recommendations=data.get("tomorrow_recommendations", []),
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )

            self._save_report(result)
            return result

        except Exception as e:
            logger.error("Post-session analysis failed: %s", e)
            return None

    def _format_trades(self, trades: list[dict], limit: int = 100) -> str:
        lines = []
        for t in trades[:limit]:
            lines.append(
                f"  [{t.get('id', '?')}] {t.get('timestamp', '?')} "
                f"{t.get('action', '?'):4s} {t.get('symbol', '?'):6s} "
                f"qty={t.get('quantity', 0)} @ ${t.get('price', 0):.2f} "
                f"| strategy={t.get('strategy', '?')} "
                f"| conviction={t.get('conviction', 0):.2f} "
                f"| pnl=${t.get('pnl', 0):+.2f}"
            )
        if len(trades) > limit:
            lines.append(f"  ... and {len(trades) - limit} more trades")
        return "\n".join(lines) or "  No trades today."

    def _format_strategy_stats(self, stats: dict) -> str:
        lines = []
        for name, s in stats.items():
            lines.append(
                f"  {name:20s} | Trades: {s.get('count', 0):3d} | "
                f"WR: {s.get('win_rate', 0):5.1f}% | "
                f"PnL: ${s.get('pnl', 0):+8.2f} | "
                f"Avg: ${s.get('avg_pnl', 0):+6.2f} | "
                f"Sharpe: {s.get('sharpe', 0):.2f}"
            )
        return "\n".join(lines) or "  No strategy data."

    def _format_orchestrator(self, decisions: list[dict]) -> str:
        lines = []
        for d in decisions[:50]:
            lines.append(
                f"  {d.get('timestamp', '?')} {d.get('symbol', '?'):6s} "
                f"| weights: {d.get('weights', {})} "
                f"| final: {d.get('action', '?')} "
                f"| confidence: {d.get('confidence', 0):.2f}"
            )
        return "\n".join(lines) or "  No orchestrator data."

    def _format_risk_events(self, events: list[dict]) -> str:
        lines = []
        for e in events:
            lines.append(
                f"  [{e.get('severity', '?')}] {e.get('timestamp', '?')} "
                f"{e.get('type', '?')}: {e.get('description', '?')}"
            )
        return "\n".join(lines) or "  No risk events."

    def _save_report(self, result: SessionAnalysis) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"report_{result.date}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": result.date,
                    "grade": result.overall_grade,
                    "findings": result.key_findings,
                    "strategy_assessments": result.strategy_assessments,
                    "parameter_suggestions": result.parameter_suggestions,
                    "regime_analysis": result.regime_analysis,
                    "recommendations": result.tomorrow_recommendations,
                    "cost_usd": result.cost_usd,
                    "latency_ms": result.latency_ms,
                },
                f,
                indent=2,
            )
        logger.info("Session report saved to %s", path)

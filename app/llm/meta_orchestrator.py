# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Weekly meta-orchestrator that adjusts strategy weights and system
parameters based on accumulated session reports and performance data.

Uses Claude for deep multi-factor reasoning. Runs weekly (Sunday) or
after N consecutive losing sessions.

auto_apply defaults to False — weight changes require human confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


META_ORCH_SYSTEM_PROMPT = """\
You are a portfolio strategist conducting a weekly review of an automated
trading system. You analyze accumulated session reports, strategy performance,
and market regime data to recommend weight adjustments and parameter changes.

CONSTRAINTS:
- Strategy weights must sum to 1.0
- No single strategy weight below 0.05 or above 0.50
- Parameter changes should be conservative (max 20% change per week)
- If a strategy has been consistently losing, reduce weight but don't zero it
- Consider regime transitions: what worked last week may not work next week

You have access to the system's strategy list:
  1. trend_following (momentum-based, works in trending markets)
  2. stat_arb_pairs (mean-reversion, works in range-bound markets)
  3. factor_model (cross-sectional, works in sector-rotation environments)
  4. pattern_trading (chart patterns, works with clear formations)

Respond ONLY with valid JSON.
"""

META_ORCH_USER_TEMPLATE = """\
WEEKLY REVIEW — Week of {week_start} to {week_end}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE SUMMARY:
  Net PnL: ${net_pnl:+,.2f} ({pnl_pct:+.2f}%)
  Total trades: {total_trades} | Win rate: {win_rate:.1f}%
  Max drawdown: {max_dd:.2f}% | Sharpe (weekly): {sharpe:.2f}

CURRENT STRATEGY WEIGHTS:
{current_weights}

DAILY SESSION GRADES:
{daily_grades}

STRATEGY PERFORMANCE (this week):
{strategy_perf}

REGIME HISTORY (this week):
{regime_history}

PREVIOUS WEEK'S RECOMMENDATIONS (if any):
{prev_recommendations}

ACCUMULATED SESSION FINDINGS (severity >= high):
{high_findings}

Respond with:
{{
  "week": "{week_start}",
  "market_assessment": "<paragraph on this week's market character>",
  "regime_forecast": "<expected regime for next week with reasoning>",
  "new_weights": {{
    "trend_following": <float>,
    "stat_arb_pairs": <float>,
    "factor_model": <float>,
    "pattern_trading": <float>
  }},
  "weight_rationale": "<why these weights>",
  "parameter_changes": [
    {{
      "path": "<config.yaml dotted path>",
      "old_value": "<current>",
      "new_value": "<recommended>",
      "rationale": "<why>"
    }}
  ],
  "risk_adjustments": {{
    "max_position_size_pct": <float or null if no change>,
    "max_daily_loss_pct": <float or null>,
    "commentary": "<risk posture recommendation>"
  }},
  "strategy_specific_notes": [
    {{
      "strategy": "<name>",
      "note": "<specific observation or tweak>"
    }}
  ],
  "confidence": <float 0.0 to 1.0>
}}
"""

_STRATEGY_KEYS = ("trend_following", "stat_arb_pairs", "factor_model", "pattern_trading")


@dataclass
class MetaOrchestratorResult:
    week: str
    market_assessment: str
    regime_forecast: str
    new_weights: dict[str, float]
    weight_rationale: str
    parameter_changes: list[dict]
    risk_adjustments: dict
    strategy_notes: list[dict]
    confidence: float
    cost_usd: float


class MetaOrchestrator:
    """
    Weekly strategy rebalancing advisor.

    Consumes accumulated PostSessionAnalyst reports and produces
    weight/parameter recommendations. Can optionally auto-apply
    changes to config.yaml with human confirmation.
    """

    def __init__(self, llm_client: LLMClient, backend: str = "claude"):
        self.llm = llm_client
        self.backend = backend

    def review(
        self,
        week_start: str,
        week_end: str,
        net_pnl: float,
        pnl_pct: float,
        total_trades: int,
        win_rate: float,
        max_dd: float,
        sharpe: float,
        current_weights: dict[str, float],
        daily_reports: list[dict],
        strategy_performance: dict,
        regime_history: list[dict],
        prev_recommendations: list[str],
    ) -> Optional[MetaOrchestratorResult]:
        """Run weekly meta-orchestration review."""
        weights_str = "\n".join(
            f"  {k}: {v:.3f}" for k, v in current_weights.items()
        )
        grades_str = "\n".join(
            f"  {r['date']}: Grade {r.get('grade', '?')} | "
            f"PnL: ${r.get('pnl', 0):+,.2f} | "
            f"Trades: {r.get('trades', 0)}"
            for r in daily_reports
        )
        perf_str = "\n".join(
            f"  {name}: PnL ${s.get('pnl', 0):+,.2f} | "
            f"WR {s.get('win_rate', 0):.1f}% | "
            f"Trades {s.get('count', 0)} | "
            f"Sharpe {s.get('sharpe', 0):.2f}"
            for name, s in strategy_performance.items()
        )
        regime_str = "\n".join(
            f"  {r['date']}: {r['regime']} (VIX: {r.get('vix', 0):.1f})"
            for r in regime_history
        )
        findings_str = "\n".join(
            f"  [{f['severity']}] {f['finding']} "
            f"(impact: ${f.get('estimated_pnl_impact', 0):+,.2f})"
            for r in daily_reports
            for f in r.get("findings", [])
            if f.get("severity") in ("critical", "high")
        ) or "  None"

        user_prompt = META_ORCH_USER_TEMPLATE.format(
            week_start=week_start,
            week_end=week_end,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            total_trades=total_trades,
            win_rate=win_rate,
            max_dd=max_dd,
            sharpe=sharpe,
            current_weights=weights_str,
            daily_grades=grades_str,
            strategy_perf=perf_str,
            regime_history=regime_str,
            prev_recommendations="\n".join(
                f"  - {r}" for r in prev_recommendations
            ) or "  None (first week)",
            high_findings=findings_str,
        )

        try:
            response = self.llm.complete(
                backend=self.backend,
                system_prompt=META_ORCH_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.2,
            )
            data = response.parse_json()

            # Validate and normalise weights
            weights = data.get("new_weights", {})
            # Clamp each weight to [0.05, 0.50]
            weights = {k: max(0.05, min(0.50, float(weights.get(k, 0.25)))) for k in _STRATEGY_KEYS}
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                weights = {k: v / total for k, v in weights.items()}

            return MetaOrchestratorResult(
                week=data.get("week", week_start),
                market_assessment=data.get("market_assessment", ""),
                regime_forecast=data.get("regime_forecast", ""),
                new_weights=weights,
                weight_rationale=data.get("weight_rationale", ""),
                parameter_changes=data.get("parameter_changes", []),
                risk_adjustments=data.get("risk_adjustments", {}),
                strategy_notes=data.get("strategy_specific_notes", []),
                confidence=float(data.get("confidence", 0.5)),
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error("Meta-orchestrator review failed: %s", e)
            return None

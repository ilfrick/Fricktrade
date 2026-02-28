# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
LLM-powered daily symbol selection.

Runs once daily before market open. Selects the trading universe from
a candidate list using Gemini's large context window, which can process
100+ candidate symbols including fundamentals and technicals in one call.

Integration: called during pre-market initialization via SymbolManager.
Falls back to rule-based filter if LLM call fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


FILTER_SYSTEM_PROMPT = """\
You are a quantitative trading universe selector for an intraday automated
trading system. Your job is to select the best symbols to trade today from
a candidate universe, based on their fundamental profile, technical setup,
recent news flow, and current market regime.

The trading system uses these strategies:
- Trend Following (works best with trending, liquid stocks)
- Statistical Arbitrage Pairs (works best with correlated pairs diverging)
- Factor Model (works best with stocks showing momentum/value/quality signals)
- Pattern Trading (works best with stocks showing clear chart patterns)

SELECTION CRITERIA (prioritize in order):
1. Liquidity: average volume > 500K shares/day, tight spreads
2. Volatility: moderate ATR — enough to profit, not so much it's untradeable
3. Catalyst presence: earnings, FDA, M&A, sector rotation, macro events
4. Strategy fit: does the current setup match at least one strategy above?
5. Avoid: binary event risk (earnings today), halted stocks, meme/squeeze dynamics

Respond ONLY with valid JSON, no markdown, no commentary.
"""

FILTER_USER_TEMPLATE = """\
Today's date: {date}
Market regime: {regime} (VIX: {vix:.1f}, SPY trend: {spy_trend})
Account equity: ${equity:,.0f}
Max simultaneous positions: {max_positions}
Currently holding: {current_positions}

Candidate universe ({n_candidates} symbols):
{candidates_block}

Select the top {select_count} symbols to trade today. For each, explain
the rationale and assign a conviction score.

Respond with this JSON:
{{
  "date": "{date}",
  "regime_assessment": "<brief market regime description>",
  "selected_symbols": [
    {{
      "symbol": "<ticker>",
      "conviction": <float 0.0 to 1.0>,
      "primary_strategy": "<trend_following|stat_arb|factor_model|pattern>",
      "rationale": "<one sentence>",
      "expected_volatility": "<low|moderate|high>",
      "catalyst": "<description or none>"
    }}
  ],
  "excluded_notable": [
    {{
      "symbol": "<ticker>",
      "reason": "<why excluded>"
    }}
  ]
}}
"""


@dataclass
class SymbolSelection:
    symbol: str
    conviction: float
    primary_strategy: str
    rationale: str
    expected_volatility: str
    catalyst: str


@dataclass
class FilterResult:
    date: str
    regime_assessment: str
    selected: list[SymbolSelection]
    excluded_notable: list[dict]
    cost_usd: float
    latency_ms: float


class DailySymbolsFilter:
    """
    Selects the trading universe once per day using Gemini.

    The large context window allows passing data for 100+ candidate
    symbols in a single call, including fundamentals and technicals.
    """

    def __init__(self, llm_client: LLMClient, backend: str = "gemini"):
        self.llm = llm_client
        self.backend = backend
        self._last_result: Optional[FilterResult] = None

    def select(
        self,
        date: str,
        candidates: list[dict],
        regime: str,
        vix: float,
        spy_trend: str,
        equity: float,
        max_positions: int,
        current_positions: list[str],
        select_count: int = 20,
    ) -> Optional[FilterResult]:
        """
        Select trading universe from candidates.

        Args:
            candidates: list of dicts with keys:
                symbol, name, sector, market_cap, avg_volume,
                atr_pct, rsi_14, adx, sma_20_trend, recent_news_count,
                top_headline, earnings_date, change_5d_pct
        """
        candidates_block = self._format_candidates(candidates)

        user_prompt = FILTER_USER_TEMPLATE.format(
            date=date,
            regime=regime,
            vix=vix,
            spy_trend=spy_trend,
            equity=equity,
            max_positions=max_positions,
            current_positions=", ".join(current_positions) or "none",
            n_candidates=len(candidates),
            candidates_block=candidates_block,
            select_count=select_count,
        )

        try:
            response = self.llm.complete(
                backend=self.backend,
                system_prompt=FILTER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.15,
            )
            data = response.parse_json()

            result = FilterResult(
                date=data["date"],
                regime_assessment=data["regime_assessment"],
                selected=[
                    SymbolSelection(**s) for s in data.get("selected_symbols", [])
                ],
                excluded_notable=data.get("excluded_notable", []),
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )
            self._last_result = result
            logger.info(
                "LLM symbol filter selected %d symbols (%.0fms, $%.4f): %s",
                len(result.selected),
                result.latency_ms,
                result.cost_usd,
                [s.symbol for s in result.selected],
            )
            return result

        except Exception as e:
            logger.error("LLM symbol filter failed: %s", e)
            return None

    def get_active_symbols(self) -> list[str]:
        """Return list of selected tickers for the trading loop."""
        if not self._last_result:
            return []
        return [s.symbol for s in self._last_result.selected]

    def get_conviction_map(self) -> dict[str, float]:
        """Return {symbol: conviction} for position sizing adjustments."""
        if not self._last_result:
            return {}
        return {s.symbol: s.conviction for s in self._last_result.selected}

    def _format_candidates(self, candidates: list[dict]) -> str:
        lines = []
        for c in candidates:
            lines.append(
                f"  {c['symbol']:6s} | {c.get('name', 'N/A'):30s} | "
                f"Sector: {c.get('sector', '?'):15s} | "
                f"AvgVol: {c.get('avg_volume', 0):>10,} | "
                f"ATR%: {c.get('atr_pct', 0):5.2f} | "
                f"RSI: {c.get('rsi_14', 50):5.1f} | "
                f"ADX: {c.get('adx', 0):5.1f} | "
                f"5d%: {c.get('change_5d_pct', 0):+6.2f} | "
                f"SMA20: {c.get('sma_20_trend', '?'):5s} | "
                f"News: {c.get('recent_news_count', 0)} | "
                f"Headline: {c.get('top_headline', 'none')[:60]}"
            )
        return "\n".join(lines)

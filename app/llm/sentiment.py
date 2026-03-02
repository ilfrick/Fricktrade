# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
LLM-powered news sentiment analysis.

Replaces the Ollama-based local model with Claude for superior
qualitative understanding of news impact on individual symbols.

Integration point: called by trader.py before signal generation.
The sentiment score is injected into the market_state dict under
the key 'llm_sentiment', which strategies can consume.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)


SENTIMENT_SYSTEM_PROMPT = """\
You are a financial news sentiment analyst for an automated trading system.
You analyze news headlines and summaries to determine their likely impact
on a stock's price in the next 1-24 hours (intraday to short-term).

RULES:
- Score from -1.0 (extremely bearish) to +1.0 (extremely bullish)
- 0.0 means neutral or irrelevant to price
- Consider second-order effects (e.g., competitor news, sector rotation)
- Account for whether the news is already priced in (earnings beats
  that match whisper numbers are neutral, not bullish)
- Be skeptical of promotional or low-quality sources

Respond ONLY with valid JSON, no markdown fences, no commentary.
"""

SENTIMENT_USER_TEMPLATE = """\
Analyze the following news items for {symbol} ({company_name}).

Current context:
- Price: ${price:.2f} | Change today: {change_pct:+.2f}%
- Sector: {sector} | Market cap: {market_cap}
- VIX: {vix:.1f} | SPY change: {spy_change:+.2f}%

News items:
{news_block}

Respond with this exact JSON structure:
{{
  "symbol": "{symbol}",
  "overall_sentiment": <float -1.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence explaining the score>",
  "catalysts": [
    {{
      "headline": "<headline>",
      "impact": <float -1.0 to 1.0>,
      "timeframe": "<immediate|hours|days>",
      "already_priced_in": <true|false>
    }}
  ],
  "recommended_bias": "<bullish|bearish|neutral>",
  "risk_flag": "<none|earnings_imminent|high_volatility_event|low_confidence>"
}}
"""


@dataclass
class SentimentResult:
    symbol: str
    overall_sentiment: float
    confidence: float
    reasoning: str
    recommended_bias: str
    risk_flag: str
    catalysts: list[dict]
    cost_usd: float
    latency_ms: float


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for trading symbols using Gemini.

    Results are cached per symbol (TTL: cache_ttl_seconds) to avoid
    redundant API calls within the same news cycle.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        backend: str = "gemini",
        cache_ttl_seconds: int = 900,
        min_confidence_to_inject: float = 0.3,
    ):
        self.llm = llm_client
        self.backend = backend
        self._cache_ttl_seconds = cache_ttl_seconds
        self._min_confidence = min_confidence_to_inject
        # {cache_key: (SentimentResult, timestamp)}
        self._cache: dict[str, tuple[SentimentResult, float]] = {}

    def analyze(
        self,
        symbol: str,
        news_items: list[dict],
        price: float,
        change_pct: float,
        company_name: str = "",
        sector: str = "Unknown",
        market_cap: str = "Unknown",
        vix: float = 20.0,
        spy_change: float = 0.0,
    ) -> Optional[SentimentResult]:
        """
        Analyze news sentiment for a single symbol.

        Args:
            news_items: list of dicts with keys:
                'headline', 'summary', 'source', 'published_at'

        Returns:
            SentimentResult or None on failure.
        """
        if not news_items:
            return SentimentResult(
                symbol=symbol,
                overall_sentiment=0.0,
                confidence=0.0,
                reasoning="No news available",
                recommended_bias="neutral",
                risk_flag="none",
                catalysts=[],
                cost_usd=0.0,
                latency_ms=0.0,
            )

        # Cache lookup
        cache_key = f"{symbol}:{hash(tuple(i.get('headline', '') for i in news_items[:5]))}"
        now = time.monotonic()
        if cache_key in self._cache:
            result, ts = self._cache[cache_key]
            if now - ts < self._cache_ttl_seconds:
                logger.debug("Sentiment cache hit for %s", symbol)
                return result

        news_block = "\n".join(
            f"  [{i + 1}] {item.get('source', 'Unknown')} "
            f"({item.get('published_at', 'N/A')})\n"
            f"      Headline: {item.get('headline', '')}\n"
            f"      Summary: {item.get('summary', 'N/A')}"
            for i, item in enumerate(news_items[:10])
        )

        user_prompt = SENTIMENT_USER_TEMPLATE.format(
            symbol=symbol,
            company_name=company_name or symbol,
            price=price,
            change_pct=change_pct,
            sector=sector,
            market_cap=market_cap,
            vix=vix,
            spy_change=spy_change,
            news_block=news_block,
        )

        try:
            response = self.llm.complete(
                backend=self.backend,
                system_prompt=SENTIMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
                temperature=0.1,
            )
            data = response.parse_json()

            result = SentimentResult(
                symbol=data.get("symbol", symbol),
                overall_sentiment=float(data["overall_sentiment"]),
                confidence=float(data["confidence"]),
                reasoning=data.get("reasoning", ""),
                recommended_bias=data.get("recommended_bias", "neutral"),
                risk_flag=data.get("risk_flag", "none"),
                catalysts=data.get("catalysts", []),
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )
            self._cache[cache_key] = (result, now)
            return result

        except Exception as e:
            logger.warning("Sentiment analysis failed for %s: %s", symbol, e)
            return None

    def inject_into_market_state(
        self,
        market_state: dict,
        result: SentimentResult,
    ) -> None:
        """
        Inject sentiment into market_state so strategies can access it
        via market_state['llm_sentiment'].

        Only injects if confidence >= min_confidence_to_inject.
        """
        if result.confidence < self._min_confidence:
            return
        market_state["llm_sentiment"] = {
            "score": result.overall_sentiment,
            "confidence": result.confidence,
            "bias": result.recommended_bias,
            "risk_flag": result.risk_flag,
        }

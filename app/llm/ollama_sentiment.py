# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Aggregate crypto-market sentiment via the configured LLM provider.

Runs periodically, reads cached news headlines, and returns a market-wide
sentiment score for entry ranking and monitoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Analyze these recent crypto/finance news headlines and rate the overall market sentiment.

Headlines:
{news_block}

Respond with ONLY a valid JSON object (no markdown, no explanation):
{{"score": <float -1.0 to 1.0>, "bias": "<bullish|bearish|neutral>", "key_theme": "<one sentence max 80 chars>", "confidence": <float 0.0 to 1.0>}}
"""


@dataclass
class MarketSentiment:
    score: float = 0.0            # -1.0 (bearish) to +1.0 (bullish)
    bias: str = "neutral"         # bullish | bearish | neutral
    key_theme: str = ""
    confidence: float = 0.0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.updated_at).total_seconds()

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "bias": self.bias,
            "key_theme": self.key_theme,
            "confidence": round(self.confidence, 2),
            "age_seconds": round(self.age_seconds()),
        }


class AggregateSentimentAnalyzer:
    """
    Provider-neutral aggregate-sentiment client.

    Uses the shared LLMClient so `llm.backends.aggregate_sentiment` can select
    Ollama, Gemini, or Claude. This also brings the call under the shared budget,
    retry policy, and usage accounting instead of making a direct unmetered HTTP
    request to Ollama.
    """

    def __init__(
        self,
        client: LLMClient,
        backend: str = "ollama",
        model: str | None = "llama3.2:3b",
        max_headlines: int = 20,
    ) -> None:
        self._client = client
        self._backend = str(backend)
        self._model = model
        self._max_headlines = max_headlines
        self._last = MarketSentiment()

    def analyze(self, headlines: list[str]) -> MarketSentiment:
        """
        Call the configured provider with up to max_headlines and cache the result.
        Returns the (possibly stale) last result on failure.
        """
        if not headlines:
            return self._last

        sample = headlines[: self._max_headlines]
        news_block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(sample))
        prompt = _PROMPT_TEMPLATE.format(news_block=news_block)

        t0 = time.monotonic()
        try:
            result = self._client.complete_json(
                backend=self._backend,
                system_prompt="Return only valid JSON matching the requested schema.",
                user_prompt=prompt,
                max_tokens=256,
                temperature=0.1,
                retries=1,
                model=self._model,
                usage="aggregate_sentiment",
            )
            bias = str(result.get("bias", "neutral")).lower()
            if bias not in {"bullish", "bearish", "neutral"}:
                bias = "neutral"
            self._last = MarketSentiment(
                score=max(-1.0, min(1.0, float(result.get("score", 0.0)))),
                bias=bias,
                key_theme=str(result.get("key_theme", ""))[:120],
                confidence=max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "Aggregate sentiment [%s/%s]: score=%.2f bias=%s conf=%.2f "
                "(%.1fs) — %s",
                self._backend, self._model, self._last.score, self._last.bias,
                self._last.confidence, elapsed, self._last.key_theme[:60],
            )
        except Exception as exc:
            logger.debug(
                "Aggregate sentiment [%s] failed (%.1fs): %s",
                self._backend, time.monotonic() - t0, exc,
            )

        return self._last

    def get_last(self) -> MarketSentiment:
        return self._last

    def to_dict(self) -> dict:
        return self._last.to_dict()


# Backwards-compatible import name. The analyzer is no longer Ollama-specific.
OllamaSentimentAnalyzer = AggregateSentimentAnalyzer

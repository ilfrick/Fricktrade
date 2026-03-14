# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Aggregate crypto market sentiment via local Ollama (llama3.2:3b).

Runs once per TMO cycle (~15 min), reads cached news headlines, returns
a market-wide sentiment score that is injected into TMO/strategic orch metrics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

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


class OllamaSentimentAnalyzer:
    """
    Lightweight aggregate-sentiment client for Ollama.

    Uses llama3.2:3b (~1.5s per call) to analyse a batch of news headlines
    and return a single market-wide sentiment score. Keeps the last result
    so callers always get a (possibly stale) answer even if Ollama is down.
    """

    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "llama3.2:3b",
        timeout: int = 30,
        max_headlines: int = 20,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_headlines = max_headlines
        self._last = MarketSentiment()

    def analyze(self, headlines: list[str]) -> MarketSentiment:
        """
        Call Ollama with up to max_headlines items and update cached sentiment.
        Returns the (possibly stale) last result on failure.
        """
        if not headlines:
            return self._last

        sample = headlines[: self._max_headlines]
        news_block = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(sample))
        prompt = _PROMPT_TEMPLATE.format(news_block=news_block)

        t0 = time.monotonic()
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 120, "temperature": 0.1},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()

            # Extract JSON even if model wraps it in markdown
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                d = json.loads(text[start:end])
                self._last = MarketSentiment(
                    score=max(-1.0, min(1.0, float(d.get("score", 0.0)))),
                    bias=str(d.get("bias", "neutral")).lower(),
                    key_theme=str(d.get("key_theme", ""))[:120],
                    confidence=max(0.0, min(1.0, float(d.get("confidence", 0.5)))),
                )
                elapsed = time.monotonic() - t0
                logger.info(
                    "Ollama sentiment: score=%.2f bias=%s conf=%.2f (%.1fs) — %s",
                    self._last.score,
                    self._last.bias,
                    self._last.confidence,
                    elapsed,
                    self._last.key_theme[:60],
                )
            else:
                logger.debug("Ollama sentiment: could not parse JSON from: %s", text[:200])
        except Exception as exc:
            logger.debug("Ollama sentiment failed (%.1fs): %s", time.monotonic() - t0, exc)

        return self._last

    def get_last(self) -> MarketSentiment:
        return self._last

    def to_dict(self) -> dict:
        return self._last.to_dict()

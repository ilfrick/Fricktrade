# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
Unified LLM client with backend-agnostic interface.
Supports Claude (Anthropic) and Gemini (Google).

All LLM interactions go through this client, which handles retries,
rate limits, cost tracking, daily budget enforcement, and structured
output parsing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    raw_response: Optional[dict] = None

    def parse_json(self) -> dict:
        """Extract JSON from response, handling markdown fences."""
        text = self.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())


@dataclass
class LLMUsageTracker:
    """Tracks cumulative cost and token usage per session."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    errors: int = 0
    _today: str = field(default_factory=lambda: str(date.today()), repr=False)

    def _reset_if_new_day(self) -> None:
        today = str(date.today())
        if today != self._today:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost_usd = 0.0
            self.call_count = 0
            self.errors = 0
            self._today = today

    def record(self, response: LLMResponse) -> None:
        self._reset_if_new_day()
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        self.call_count += 1

    def daily_cost(self) -> float:
        self._reset_if_new_day()
        return self.total_cost_usd

    def summary(self) -> dict:
        self._reset_if_new_day()
        return {
            "calls": self.call_count,
            "errors": self.errors,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
        }


class LLMBackend(ABC):
    """Abstract backend interface."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


class ClaudeBackend(LLMBackend):
    """Anthropic Claude backend."""

    # Pricing per 1M tokens (Sonnet 4.6 as of Feb 2026)
    INPUT_COST_PER_M = 3.00
    OUTPUT_COST_PER_M = 15.00

    def __init__(self, model: Optional[str] = None):
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        self.model = model or os.getenv(
            "LLM_CLAUDE_MODEL", "claude-sonnet-4-6"
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> LLMResponse:
        t0 = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency = (time.monotonic() - t0) * 1000

        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        cost = (
            input_tok * self.INPUT_COST_PER_M / 1_000_000
            + output_tok * self.OUTPUT_COST_PER_M / 1_000_000
        )

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=round(latency, 1),
            cost_usd=cost,
            raw_response=response.model_dump(),
        )


class GeminiBackend(LLMBackend):
    """Google Gemini backend."""

    # Pricing per 1M tokens (Gemini 2.5 Flash as of Feb 2026, thinking disabled)
    INPUT_COST_PER_M = 0.075
    OUTPUT_COST_PER_M = 0.30

    def __init__(self, model: Optional[str] = None):
        from google import genai
        self.client = genai.Client(
            api_key=os.environ["GOOGLE_GEMINI_API_KEY"]
        )
        self.model = model or os.getenv(
            "LLM_GEMINI_MODEL", "gemini-2.5-flash"
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> LLMResponse:
        t0 = time.monotonic()
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        latency = (time.monotonic() - t0) * 1000

        input_tok = response.usage_metadata.prompt_token_count or 0
        output_tok = response.usage_metadata.candidates_token_count or 0
        cost = (
            input_tok * self.INPUT_COST_PER_M / 1_000_000
            + output_tok * self.OUTPUT_COST_PER_M / 1_000_000
        )

        # response.text can be None when thinking consumes all tokens; fall back to parts
        content = response.text
        if content is None:
            for candidate in (response.candidates or []):
                for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                    if getattr(part, "text", None):
                        content = part.text
                        break
                if content is not None:
                    break

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=round(latency, 1),
            cost_usd=cost,
            raw_response=None,
        )


class LLMClient:
    """
    Main client used by all Fricktrade LLM modules.

    Routes requests to the appropriate backend with retry logic,
    cost tracking, and a daily budget circuit breaker.
    """

    BACKENDS: dict[str, type[LLMBackend]] = {
        "claude": ClaudeBackend,
        "gemini": GeminiBackend,
    }

    def __init__(self, daily_budget_usd: float = 5.00, alert_threshold_usd: float = 4.00):
        self._backends: dict[str, LLMBackend] = {}
        self._trackers: dict[str, LLMUsageTracker] = {}
        self._daily_budget_usd = daily_budget_usd
        self._alert_threshold_usd = alert_threshold_usd

    def _get_backend(self, name: str) -> LLMBackend:
        if name not in self._backends:
            if name not in self.BACKENDS:
                raise ValueError(f"Unknown LLM backend: {name}")
            self._backends[name] = self.BACKENDS[name]()
            self._trackers[name] = LLMUsageTracker()
        return self._backends[name]

    def _daily_total_cost(self) -> float:
        return sum(t.daily_cost() for t in self._trackers.values())

    def complete(
        self,
        backend: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        retries: int = 3,
        retry_delay: float = 2.0,
        critical: bool = False,
    ) -> LLMResponse:
        """
        Send a completion request with automatic retries.

        Args:
            critical: If True, bypasses the daily budget circuit breaker.
                      Use for risk-critical calls only.
        """
        b = self._get_backend(backend)
        tracker = self._trackers[backend]

        # Daily budget circuit breaker
        daily_cost = self._daily_total_cost()
        if not critical and daily_cost >= self._daily_budget_usd:
            raise RuntimeError(
                f"LLM daily budget exhausted: ${daily_cost:.4f} >= ${self._daily_budget_usd:.2f}"
            )
        if daily_cost >= self._alert_threshold_usd:
            logger.warning(
                "LLM daily cost approaching budget: $%.4f / $%.2f",
                daily_cost,
                self._daily_budget_usd,
            )

        delay = retry_delay
        for attempt in range(retries):
            try:
                response = b.complete(
                    system_prompt, user_prompt, max_tokens, temperature
                )
                tracker.record(response)
                logger.info(
                    "LLM [%s] %d in/%d out, %.0fms, $%.4f (daily: $%.4f)",
                    backend,
                    response.input_tokens,
                    response.output_tokens,
                    response.latency_ms,
                    response.cost_usd,
                    self._daily_total_cost(),
                )
                return response
            except RuntimeError:
                raise
            except Exception as e:
                tracker.errors += 1
                if attempt < retries - 1:
                    logger.warning(
                        "LLM [%s] attempt %d failed: %s. Retrying in %.1fs",
                        backend,
                        attempt + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error("LLM [%s] all retries exhausted: %s", backend, e)
                    raise

    def complete_json(
        self,
        backend: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> dict:
        """Complete and parse JSON response."""
        response = self.complete(backend, system_prompt, user_prompt, **kwargs)
        return response.parse_json()

    def usage_summary(self) -> dict:
        return {name: t.summary() for name, t in self._trackers.items()}

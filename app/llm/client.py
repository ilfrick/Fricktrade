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
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
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
        text = (self.content or "").strip()
        if not text:
            raise ValueError("Empty LLM response")
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

    def __post_init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()

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
        with self._lock:
            self._reset_if_new_day()
            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens
            self.total_cost_usd += response.cost_usd
            self.call_count += 1

    def daily_cost(self) -> float:
        with self._lock:
            self._reset_if_new_day()
            return self.total_cost_usd

    def summary(self) -> dict:
        with self._lock:
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
        model: Optional[str] = None,
    ) -> LLMResponse: ...


class ClaudeBackend(LLMBackend):
    """Anthropic Claude backend."""

    # Pricing per 1M tokens (Sonnet 4.6 as of Feb 2026)
    INPUT_COST_PER_M = 3.00
    OUTPUT_COST_PER_M = 15.00

    def __init__(self, model: Optional[str] = None):
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            default_headers={"Referer": "www.housefz.com"},
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
        model: Optional[str] = None,
    ) -> LLMResponse:
        t0 = time.monotonic()
        effective_model = model or self.model
        response = self.client.messages.create(
            model=effective_model,
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
            model=effective_model,
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
            api_key=os.environ["GOOGLE_GEMINI_API_KEY"],
            http_options={"headers": {"Referer": "https://www.housefz.com"}},
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
        model: Optional[str] = None,
    ) -> LLMResponse:
        t0 = time.monotonic()
        effective_model = model or self.model
        # gemini-2.5-pro supports thinking; disabling it (budget=0) is rejected
        use_thinking = "pro" in effective_model
        thinking_cfg = {} if use_thinking else {"thinking_config": {"thinking_budget": 0}}
        response = self.client.models.generate_content(
            model=effective_model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                **thinking_cfg,
            },
        )
        latency = (time.monotonic() - t0) * 1000

        input_tok = response.usage_metadata.prompt_token_count or 0
        output_tok = response.usage_metadata.candidates_token_count or 0
        cost = (
            input_tok * self.INPUT_COST_PER_M / 1_000_000
            + output_tok * self.OUTPUT_COST_PER_M / 1_000_000
        )

        # response.text can be None when thinking consumes all tokens; fall back to parts.
        # Skip thought=True parts (internal reasoning) — only take the actual response part.
        content = response.text
        if not content:
            for candidate in (response.candidates or []):
                for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
                    if getattr(part, "text", None) and not getattr(part, "thought", False):
                        content = part.text
                        break
                if content:
                    break
        if not content:
            content = ""

        return LLMResponse(
            content=content,
            model=effective_model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=round(latency, 1),
            cost_usd=cost,
            raw_response=None,
        )


class OllamaBackend(LLMBackend):
    """Local Ollama backend (free — no API cost).

    Uses a persistent ThreadPoolExecutor with future.result(timeout=45) as a
    hard wall-clock deadline. urllib.request timeout=30 only fires on idle
    socket reads; trickle responses from a slow model bypass it entirely —
    the same lesson learned with the Binance demo API deadlocks.
    """

    _WALL_CLOCK_TIMEOUT = 45  # seconds; hard kill regardless of response streaming

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("LLM_OLLAMA_MODEL", "llama3.1:8b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        # Persistent pool — intentionally NOT used as context manager so __exit__
        # doesn't call shutdown(wait=True) and block past the timeout.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ollama")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        model: Optional[str] = None,
    ) -> LLMResponse:
        from concurrent.futures import TimeoutError as _FutTimeoutError
        import urllib.request
        effective_model = model or self.model

        def _call() -> tuple[dict, float]:
            t0 = time.monotonic()
            payload = json.dumps({
                "model": effective_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read()), time.monotonic() - t0

        future = self._executor.submit(_call)
        try:
            data, elapsed = future.result(timeout=self._WALL_CLOCK_TIMEOUT)
        except _FutTimeoutError:
            raise RuntimeError(f"Ollama request timed out after {self._WALL_CLOCK_TIMEOUT}s")
        latency = elapsed * 1000
        content = data.get("message", {}).get("content", "") or ""
        prompt_tok = data.get("prompt_eval_count", 0) or 0
        eval_tok = data.get("eval_count", 0) or 0
        return LLMResponse(
            content=content,
            model=effective_model,
            input_tokens=prompt_tok,
            output_tokens=eval_tok,
            latency_ms=round(latency, 1),
            cost_usd=0.0,
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
        "ollama": OllamaBackend,
    }

    def __init__(self, daily_budget_usd: float = 5.00, alert_threshold_usd: float = 4.00):
        self._backends: dict[str, LLMBackend] = {}
        self._trackers: dict[str, LLMUsageTracker] = {}
        self._daily_budget_usd = daily_budget_usd
        self._alert_threshold_usd = alert_threshold_usd
        self._backend_lock = threading.Lock()

    def _get_backend(self, name: str) -> LLMBackend:
        with self._backend_lock:
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
        model: Optional[str] = None,
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

        # Local backends don't benefit from retries — they're either available or not.
        # Cap at 1 attempt to prevent cycle stalls when Ollama is slow/loading.
        if backend == "ollama":
            retries = 1
        delay = retry_delay
        for attempt in range(retries):
            try:
                response = b.complete(
                    system_prompt, user_prompt, max_tokens, temperature, model=model
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

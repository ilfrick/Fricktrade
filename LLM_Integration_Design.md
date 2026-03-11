# Fricktrade 3.0 — LLM Integration Design

**Author**: Claude (Anthropic) + Nicola  
**Date**: February 28, 2026  
**Branch**: 3.0  
**Status**: Design Document

---

## 1. Overview

This document describes the integration of Claude (Anthropic) and Gemini (Google) into the Fricktrade 3.0 trading system. The LLMs operate on **slow cycles** (daily, per-session, per-event) and feed parameters/configurations to the real-time system. They are never placed in the critical execution path of a trade.

### Architecture Principle

```
┌─────────────────────────────────────────────────────────┐
│                   LLM LAYER (async)                     │
│                                                         │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │  Symbols    │ │   News /     │ │  Post-Session    │  │
│  │  Filter     │ │  Sentiment   │ │  Analyst         │  │
│  │  (daily)    │ │  (pre-bar)   │ │  (end of day)    │  │
│  │  [Gemini]   │ │  [Claude]    │ │  [Gemini]        │  │
│  └──────┬──────┘ └──────┬───────┘ └────────┬─────────┘  │
│         │               │                  │            │
│  ┌──────┴───────────────┴──────────────────┴─────────┐  │
│  │          Meta-Orchestrator (weekly) [Claude]       │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────┴────────────────────────────┐  │
│  │     Risk Event Interpreter (on-demand) [Claude]   │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │  (writes to config / cache / DB)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FRICKTRADE 3.0 (real-time loop)            │
│                                                         │
│   Data → Features → Strategies → Orchestrator → Risk    │
│                                           → Executor    │
└─────────────────────────────────────────────────────────┘
```

### Required API Keys

Add the following to `.env` in the project root:

```env
# Anthropic Claude API
ANTHROPIC_API_KEY=sk-ant-api03-...

# Google Gemini API
GOOGLE_GEMINI_API_KEY=AIza...

# Optional: override default models
LLM_CLAUDE_MODEL=claude-sonnet-4-20250514
LLM_GEMINI_MODEL=gemini-2.5-pro
```

### Required Dependencies

Add to `requirements.txt`:

```
anthropic>=0.42.0
google-genai>=1.0.0
```

---

## 2. Core Abstraction: LLMClient

All LLM interactions go through a unified client that handles retries, rate limits, cost tracking, and structured output parsing.

### File: `app/llm/client.py`

```python
"""
Unified LLM client with backend-agnostic interface.
Supports Claude (Anthropic) and Gemini (Google).
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
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
        return json.loads(text)


@dataclass
class LLMUsageTracker:
    """Tracks cumulative cost and token usage per session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    errors: int = 0

    def record(self, response: LLMResponse):
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        self.call_count += 1

    def summary(self) -> dict:
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
    ) -> LLMResponse:
        ...


class ClaudeBackend(LLMBackend):
    """Anthropic Claude backend."""

    # Pricing per 1M tokens (Sonnet 4 as of Feb 2026)
    INPUT_COST_PER_M = 3.00
    OUTPUT_COST_PER_M = 15.00

    def __init__(self, model: Optional[str] = None):
        import anthropic
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        self.model = model or os.getenv(
            "LLM_CLAUDE_MODEL", "claude-sonnet-4-20250514"
        )

    def complete(self, system_prompt, user_prompt, max_tokens=4096,
                 temperature=0.2) -> LLMResponse:
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

    # Pricing per 1M tokens (Gemini 2.5 Pro as of Feb 2026)
    INPUT_COST_PER_M = 1.25
    OUTPUT_COST_PER_M = 10.00

    def __init__(self, model: Optional[str] = None):
        from google import genai
        self.client = genai.Client(
            api_key=os.environ["GOOGLE_GEMINI_API_KEY"]
        )
        self.model = model or os.getenv(
            "LLM_GEMINI_MODEL", "gemini-2.5-pro"
        )

    def complete(self, system_prompt, user_prompt, max_tokens=4096,
                 temperature=0.2) -> LLMResponse:
        t0 = time.monotonic()
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        latency = (time.monotonic() - t0) * 1000

        input_tok = response.usage_metadata.prompt_token_count or 0
        output_tok = response.usage_metadata.candidates_token_count or 0
        cost = (
            input_tok * self.INPUT_COST_PER_M / 1_000_000
            + output_tok * self.OUTPUT_COST_PER_M / 1_000_000
        )

        return LLMResponse(
            content=response.text,
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
    Routes requests to the appropriate backend with retry logic.
    """

    BACKENDS = {
        "claude": ClaudeBackend,
        "gemini": GeminiBackend,
    }

    def __init__(self):
        self._backends: dict[str, LLMBackend] = {}
        self._trackers: dict[str, LLMUsageTracker] = {}

    def _get_backend(self, name: str) -> LLMBackend:
        if name not in self._backends:
            if name not in self.BACKENDS:
                raise ValueError(f"Unknown LLM backend: {name}")
            self._backends[name] = self.BACKENDS[name]()
            self._trackers[name] = LLMUsageTracker()
        return self._backends[name]

    def complete(
        self,
        backend: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> LLMResponse:
        """Send a completion request with automatic retries."""
        b = self._get_backend(backend)
        tracker = self._trackers[backend]

        for attempt in range(retries):
            try:
                response = b.complete(
                    system_prompt, user_prompt, max_tokens, temperature
                )
                tracker.record(response)
                logger.info(
                    "LLM [%s] %d in/%d out, %.0fms, $%.4f",
                    backend,
                    response.input_tokens,
                    response.output_tokens,
                    response.latency_ms,
                    response.cost_usd,
                )
                return response
            except Exception as e:
                tracker.errors += 1
                if attempt < retries - 1:
                    logger.warning(
                        "LLM [%s] attempt %d failed: %s. Retrying in %.1fs",
                        backend, attempt + 1, e, retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("LLM [%s] all retries exhausted: %s",
                                 backend, e)
                    raise

    def complete_json(self, backend: str, system_prompt: str,
                      user_prompt: str, **kwargs) -> dict:
        """Complete and parse JSON response."""
        response = self.complete(backend, system_prompt, user_prompt, **kwargs)
        return response.parse_json()

    def usage_summary(self) -> dict:
        return {name: t.summary() for name, t in self._trackers.items()}
```

---

## 3. Module 1: News / Sentiment Analyzer

**Backend**: Claude (superior qualitative reasoning)  
**Trigger**: Before each bar cycle, or on news event  
**Replaces**: Current Ollama-based local LLM sentiment analysis

### File: `app/llm/sentiment.py`

```python
"""
LLM-powered news sentiment analysis.
Replaces the Ollama-based local model with Claude for superior
qualitative understanding of news impact on individual symbols.
"""

import logging
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
    Analyzes news sentiment for trading symbols using Claude.

    Integration point: called by trader.py before signal generation.
    The sentiment score is injected into the market_state dict
    under the key 'llm_sentiment', which strategies can consume.
    """

    def __init__(self, llm_client: LLMClient, backend: str = "claude"):
        self.llm = llm_client
        self.backend = backend
        self._cache: dict[str, SentimentResult] = {}
        self._cache_ttl_seconds = 900  # 15 min cache per symbol

    def analyze(
        self,
        symbol: str,
        company_name: str,
        news_items: list[dict],
        price: float,
        change_pct: float,
        sector: str,
        market_cap: str,
        vix: float,
        spy_change: float,
    ) -> Optional[SentimentResult]:
        """
        Analyze news sentiment for a single symbol.

        Args:
            news_items: list of dicts with keys 'headline', 'summary',
                        'source', 'published_at'

        Returns:
            SentimentResult or None on failure.
        """
        if not news_items:
            return SentimentResult(
                symbol=symbol, overall_sentiment=0.0, confidence=0.0,
                reasoning="No news available", recommended_bias="neutral",
                risk_flag="none", catalysts=[], cost_usd=0.0, latency_ms=0.0,
            )

        # Check cache
        cache_key = f"{symbol}:{hash(str(news_items))}"
        if cache_key in self._cache:
            logger.debug("Sentiment cache hit for %s", symbol)
            return self._cache[cache_key]

        # Format news block
        news_block = "\n".join(
            f"  [{i+1}] {item.get('source', 'Unknown')} "
            f"({item.get('published_at', 'N/A')})\n"
            f"      Headline: {item['headline']}\n"
            f"      Summary: {item.get('summary', 'N/A')}"
            for i, item in enumerate(news_items[:10])  # Limit to 10 items
        )

        user_prompt = SENTIMENT_USER_TEMPLATE.format(
            symbol=symbol,
            company_name=company_name,
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
                symbol=data["symbol"],
                overall_sentiment=float(data["overall_sentiment"]),
                confidence=float(data["confidence"]),
                reasoning=data["reasoning"],
                recommended_bias=data["recommended_bias"],
                risk_flag=data.get("risk_flag", "none"),
                catalysts=data.get("catalysts", []),
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error("Sentiment analysis failed for %s: %s", symbol, e)
            return None

    def inject_into_market_state(
        self, market_state: dict, result: SentimentResult
    ):
        """
        Inject sentiment into the market_state dict so strategies
        can access it via market_state['llm_sentiment'].
        """
        market_state["llm_sentiment"] = {
            "score": result.overall_sentiment,
            "confidence": result.confidence,
            "bias": result.recommended_bias,
            "risk_flag": result.risk_flag,
        }
```

### Integration into `trader.py`

In the `_prepare_market_state()` method or equivalent, after fetching news:

```python
# In trader.py __init__:
from app.llm.client import LLMClient
from app.llm.sentiment import NewsSentimentAnalyzer

self._llm_client = LLMClient()
self._sentiment_analyzer = NewsSentimentAnalyzer(self._llm_client)

# In _prepare_market_state() or _process_single_symbol():
if news_items and self._config.get("llm_sentiment_enabled", False):
    sentiment = self._sentiment_analyzer.analyze(
        symbol=symbol,
        company_name=company_info.get("name", symbol),
        news_items=news_items,
        price=current_price,
        change_pct=daily_change_pct,
        sector=company_info.get("sector", "Unknown"),
        market_cap=company_info.get("market_cap", "Unknown"),
        vix=self._vix_value,
        spy_change=self._spy_change,
    )
    if sentiment:
        self._sentiment_analyzer.inject_into_market_state(
            market_state, sentiment
        )
```

---

## 4. Module 2: Daily Symbols Filter

**Backend**: Gemini (large context window for many symbols, lower cost)  
**Trigger**: Once daily, before market open (or pre-session)  
**Replaces**: PPO-based symbol selector that fails to converge

### File: `app/llm/symbols_filter.py`

```python
"""
LLM-powered daily symbol selection.
Replaces the PPO-based AI symbol filter with a reasoning-based approach
that works immediately without training.
"""

import json
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
                    SymbolSelection(**s) for s in data["selected_symbols"]
                ],
                excluded_notable=data.get("excluded_notable", []),
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )
            self._last_result = result
            logger.info(
                "Symbol filter selected %d symbols (%.0fms, $%.4f)",
                len(result.selected), result.latency_ms, result.cost_usd,
            )
            return result

        except Exception as e:
            logger.error("Symbol filter failed: %s", e)
            return None

    def get_active_symbols(self) -> list[str]:
        """Return list of selected tickers for the trading loop."""
        if not self._last_result:
            return []
        return [s.symbol for s in self._last_result.selected]

    def get_conviction_map(self) -> dict[str, float]:
        """Return symbol -> conviction mapping for position sizing."""
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
```

### Integration into `trader.py`

Call during the pre-market initialization phase:

```python
# In the daily init / pre-market routine:
from app.llm.symbols_filter import DailySymbolsFilter

self._symbols_filter = DailySymbolsFilter(self._llm_client)

def _init_daily_universe(self):
    """Run once at session start to select today's trading universe."""
    candidates = self._build_candidate_data()  # existing universe logic
    result = self._symbols_filter.select(
        date=today_str,
        candidates=candidates,
        regime=self._current_regime,
        vix=self._vix_value,
        spy_trend=self._spy_trend_label,
        equity=self._account_equity,
        max_positions=self._config["max_positions"],
        current_positions=[p.symbol for p in self._open_positions],
        select_count=self._config.get("llm_select_count", 20),
    )
    if result:
        self._active_symbols = self._symbols_filter.get_active_symbols()
        self._conviction_map = self._symbols_filter.get_conviction_map()
        logger.info("Daily universe: %s", self._active_symbols)
    else:
        logger.warning("LLM filter failed, falling back to rule-based filter")
        self._active_symbols = self._fallback_symbol_selection()
```

---

## 5. Module 3: Post-Session Analyst

**Backend**: Gemini (large context for full trade log)  
**Trigger**: End of trading session  
**Purpose**: Automated daily review of trading performance and strategy effectiveness

### File: `app/llm/post_session.py`

```python
"""
Post-session trade analysis using LLM.
Runs at end of each trading day to identify patterns, errors,
and suggest adjustments. Replaces manual daily review.
"""

import json
import logging
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

    Output is saved to disk and can optionally auto-apply weight
    adjustments to the orchestrator config.
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
            trade_log=self._format_trades(trades),
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
                key_findings=data["key_findings"],
                strategy_assessments=data["strategy_assessments"],
                parameter_suggestions=data["parameter_suggestions"],
                regime_analysis=data["regime_analysis"],
                tomorrow_recommendations=data["tomorrow_recommendations"],
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
            )

            self._save_report(result)
            return result

        except Exception as e:
            logger.error("Post-session analysis failed: %s", e)
            return None

    def _format_trades(self, trades: list[dict]) -> str:
        lines = []
        for t in trades[:100]:  # Limit to 100 trades for context window
            lines.append(
                f"  [{t.get('id', '?')}] {t.get('timestamp', '?')} "
                f"{t.get('action', '?'):4s} {t.get('symbol', '?'):6s} "
                f"qty={t.get('quantity', 0)} @ ${t.get('price', 0):.2f} "
                f"| strategy={t.get('strategy', '?')} "
                f"| conviction={t.get('conviction', 0):.2f} "
                f"| pnl=${t.get('pnl', 0):+.2f}"
            )
        if len(trades) > 100:
            lines.append(f"  ... and {len(trades) - 100} more trades")
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

    def _save_report(self, result: SessionAnalysis):
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"report_{result.date}.json")
        with open(path, "w") as f:
            json.dump({
                "date": result.date,
                "grade": result.overall_grade,
                "findings": result.key_findings,
                "strategy_assessments": result.strategy_assessments,
                "parameter_suggestions": result.parameter_suggestions,
                "regime_analysis": result.regime_analysis,
                "recommendations": result.tomorrow_recommendations,
                "cost_usd": result.cost_usd,
            }, f, indent=2)
        logger.info("Session report saved to %s", path)
```

---

## 6. Module 4: Meta-Orchestrator

**Backend**: Claude (complex multi-factor reasoning)  
**Trigger**: Weekly, or after N consecutive losing sessions  
**Purpose**: Adjust strategy weights and parameters based on accumulated evidence

### File: `app/llm/meta_orchestrator.py`

```python
"""
Weekly meta-orchestrator that adjusts strategy weights and system
parameters based on accumulated session reports and performance data.
Uses Claude for deep multi-factor reasoning.
"""

import json
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

        # Format inputs
        weights_str = "\n".join(
            f"  {k}: {v:.3f}" for k, v in current_weights.items()
        )
        grades_str = "\n".join(
            f"  {r['date']}: Grade {r['grade']} | "
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

            # Validate weights sum to 1.0
            weights = data["new_weights"]
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                weights = {k: v / total for k, v in weights.items()}

            return MetaOrchestratorResult(
                week=data["week"],
                market_assessment=data["market_assessment"],
                regime_forecast=data["regime_forecast"],
                new_weights=weights,
                weight_rationale=data["weight_rationale"],
                parameter_changes=data.get("parameter_changes", []),
                risk_adjustments=data.get("risk_adjustments", {}),
                strategy_notes=data.get("strategy_specific_notes", []),
                confidence=data.get("confidence", 0.5),
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error("Meta-orchestrator review failed: %s", e)
            return None
```

---

## 7. Module 5: Risk Event Interpreter

**Backend**: Claude (critical reasoning quality)  
**Trigger**: When drift monitor or risk manager fires an alert  
**Purpose**: Decide if an anomaly is a real structural break or noise

### File: `app/llm/risk_interpreter.py`

```python
"""
LLM-powered risk event interpretation.
When the drift monitor or risk manager detects an anomaly, this module
provides qualitative analysis to decide whether it's a real structural
break or transient noise.
"""

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

    Integrates with the existing drift monitor and risk manager
    to provide intelligent triage instead of simple auto-rollback.
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
        feature_drift_z: float,
        pnl_drift_z: float,
        rolling_sharpe: float,
        consecutive_losses: int,
        hmm_state: str,
        vix: float,
        vix_change: float,
        spy_change: float,
        sector_rotation: str,
        recent_performance: list[dict],
        positions: list[dict],
        model_version: str,
        best_model_version: str,
        rollback_available: bool,
    ) -> Optional[RiskInterpretation]:
        """Interpret a risk alert with full context."""

        recent_str = "\n".join(
            f"  {p['date']}: PnL ${p.get('pnl', 0):+,.2f} | "
            f"Trades: {p.get('trades', 0)} | WR: {p.get('win_rate', 0):.0f}%"
            for p in recent_performance
        )
        positions_str = "\n".join(
            f"  {p['symbol']:6s} {p['side']:5s} qty={p['quantity']} "
            f"entry=${p['entry_price']:.2f} "
            f"current=${p['current_price']:.2f} "
            f"unrealized=${p.get('unrealized_pnl', 0):+,.2f}"
            for p in positions
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
                temperature=0.1,  # Low temp for risk decisions
            )
            data = response.parse_json()

            return RiskInterpretation(
                diagnosis=data["diagnosis"],
                confidence=data["confidence"],
                explanation=data["explanation"],
                likely_cause=data["likely_cause"],
                recommended_action=data["recommended_action"],
                position_action=data["position_action"],
                should_rollback_model=data["should_rollback_model"],
                rollback_rationale=data["rollback_rationale"],
                resume_conditions=data["resume_conditions"],
                urgency=data["urgency"],
                cost_usd=response.cost_usd,
            )

        except Exception as e:
            logger.error("Risk interpretation failed: %s", e)
            return None
```

---

## 8. Configuration

### Addition to `config/config.yaml`

```yaml
llm:
  enabled: true

  # Backend assignment per module
  backends:
    sentiment: claude        # Quality-critical
    symbols_filter: gemini   # High volume, large context
    post_session: gemini     # Large trade logs
    meta_orchestrator: claude # Complex reasoning
    risk_interpreter: claude  # Safety-critical

  # Sentiment analyzer
  sentiment:
    enabled: true
    cache_ttl_seconds: 900
    max_news_items: 10
    min_confidence_to_inject: 0.3

  # Daily symbols filter
  symbols_filter:
    enabled: true
    run_time: "pre_market"   # pre_market | manual
    select_count: 20
    fallback_to_rules: true  # Use rule-based filter if LLM fails

  # Post-session analyst
  post_session:
    enabled: true
    auto_run: true           # Run automatically at session end
    output_dir: "data/session_reports"
    max_trades_in_context: 100
    auto_apply_weights: false  # If true, apply weight changes without confirmation

  # Meta-orchestrator
  meta_orchestrator:
    enabled: true
    run_day: "sunday"        # Day of week for weekly review
    auto_apply: false        # Require human confirmation
    max_weight_change: 0.15  # Max change per strategy per week
    min_sessions_required: 3 # Minimum sessions before adjusting

  # Risk event interpreter
  risk_interpreter:
    enabled: true
    min_severity: "high"     # Only interpret high/critical alerts
    timeout_seconds: 10      # Max wait for LLM response

  # Cost management
  cost:
    daily_budget_usd: 5.00
    alert_threshold_usd: 4.00
    log_all_costs: true
```

---

## 9. File Structure

```
app/
├── llm/
│   ├── __init__.py
│   ├── client.py              # LLMClient, backends, usage tracking
│   ├── sentiment.py           # News sentiment analyzer (Claude)
│   ├── symbols_filter.py      # Daily symbol selection (Gemini)
│   ├── post_session.py        # End-of-day trade analysis (Gemini)
│   ├── meta_orchestrator.py   # Weekly weight rebalancing (Claude)
│   └── risk_interpreter.py    # Risk event triage (Claude)
├── agents/
│   ├── trader.py              # Modified: integrates LLM modules
│   └── orchestrator.py        # Modified: reads LLM weight suggestions
├── ...
config/
├── config.yaml                # Modified: llm section added
data/
├── session_reports/           # New: daily JSON reports
│   ├── report_2026-02-28.json
│   └── ...
```

---

## 10. Estimated Daily Cost

| Module              | Backend | Calls/day | Avg tokens | Est. cost |
|---------------------|---------|-----------|------------|-----------|
| Sentiment analyzer  | Claude  | ~30       | ~2K in/1K out | $0.63  |
| Symbols filter      | Gemini  | 1         | ~8K in/2K out | $0.03  |
| Post-session analyst| Gemini  | 1         | ~15K in/4K out| $0.06  |
| Risk interpreter    | Claude  | ~2        | ~3K in/1K out | $0.05  |
| **Daily total**     |         |           |            | **~$0.77** |
| Meta-orchestrator   | Claude  | 1/week    | ~5K in/2K out | $0.05/wk |

**Monthly estimate: ~$17–25** depending on trading activity and alert frequency.

---

## 11. Implementation Priority

| Priority | Module             | Effort  | Expected Impact |
|----------|--------------------|---------|-----------------|
| 1        | `client.py`        | 1 day   | Foundation       |
| 2        | `sentiment.py`     | 1 day   | High — immediate signal quality improvement |
| 3        | `post_session.py`  | 1-2 days| High — automated feedback loop |
| 4        | `symbols_filter.py`| 1 day   | Medium — replaces broken PPO filter |
| 5        | `risk_interpreter.py`| 1 day | Medium — smarter than auto-rollback |
| 6        | `meta_orchestrator.py`| 1 day | Medium — weekly weight tuning |

**Total estimated effort: 6-8 days**

---

## 12. Safety Guardrails

1. **LLM never executes trades directly.** It produces scores, weights, and recommendations that the existing deterministic pipeline consumes.

2. **Fallback on failure.** Every module has a fallback path — if the LLM call fails or times out, the system continues with its existing logic (rule-based filter, current weights, auto-rollback).

3. **Cost circuit breaker.** The `LLMUsageTracker` enforces the daily budget. Once `daily_budget_usd` is exceeded, all non-critical LLM calls are skipped.

4. **Human-in-the-loop for weight changes.** `auto_apply` defaults to `false` for both `post_session` and `meta_orchestrator`. Weight and parameter changes are logged and require manual confirmation until trust is established.

5. **Deterministic audit trail.** Every LLM call is logged with full prompt, response, cost, and latency. Session reports are persisted as JSON for backtesting the LLM's own accuracy over time.

6. **No LLM in the critical path.** The real-time trading loop (bar processing → signal generation → orchestration → execution) never blocks on an LLM call. All LLM modules operate asynchronously on slower cycles.

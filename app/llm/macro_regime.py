# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
LLM-powered macro regime analyzer.

Fetches key macro indicators (VIX, 10Y yield, DXY) from FRED, optionally
combines with recent news headlines, and asks Claude to classify the current
macro regime into one of:  risk_on / risk_off / rotation / range_bound / crisis

Includes a 4-hour TTL cache so it fires at most every refresh_hours even when
called per-symbol in the hot path.
"""

from __future__ import annotations

import logging
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

MACRO_SYSTEM_PROMPT = """\
You are a macro-economic regime classifier for a quantitative trading system.
Given key macro indicators and optionally recent news headlines, classify the
current macro regime and recommend strategy weight adjustments.

Respond ONLY with valid JSON — no prose outside the JSON.
"""

MACRO_USER_TEMPLATE = """\
MACRO INDICATORS ({ts})
━━━━━━━━━━━━━━━━━━━━━━━━━━
VIX (fear index):     {vix}
10Y Treasury yield:   {dgs10}%
DXY (USD index):      {dxy}

RECENT NEWS HEADLINES:
{headlines}

Classify the macro regime. Available regimes:
  risk_on        - equity-friendly: low VIX, stable/falling yields, weak USD
  risk_off       - defensive: rising VIX, flight to safety, strong USD
  rotation       - sector rotation underway; mixed signals
  range_bound    - low-conviction sideways; low volatility
  crisis         - VIX >30, sharp yield/FX moves, tail-risk elevated

Respond with:
{{
  "name": "<regime>",
  "confidence": <float 0.0-1.0>,
  "rationale": "<1-2 sentence explanation>",
  "weight_overrides": {{
    "<strategy_name>": <multiplier float>,
    ...
  }}
}}

Include only weight overrides for strategies that should change. Strategy names:
trend_following, factor_model, pattern_trading, stat_arb_pairs, crypto_momentum,
crypto_mean_reversion, gap_reversal, earnings_drift.
"""

# FRED series: latest observation value
FRED_SERIES = {
    "vix": "VIXCLS",
    "dgs10": "DGS10",
    "dxy": "DTWEXBGS",
}


@dataclass
class MacroRegime:
    name: str  # risk_on / risk_off / rotation / range_bound / crisis
    confidence: float
    weight_overrides: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MacroRegimeAnalyzer:
    """Periodically fetches macro data and classifies market regime via LLM."""

    def __init__(self, client: LLMClient, cfg: dict):
        self._client = client
        self._cfg = cfg
        self._fred_api_key = str(cfg.get("fred_api_key", "") or "")
        self._fred_base_url = str(cfg.get("fred_base_url",
                                          "https://api.stlouisfed.org/fred") or "")
        self._refresh_hours = int(cfg.get("refresh_hours", 4))
        self._backend = str(cfg.get("backend", "claude"))
        self._cache: Optional[MacroRegime] = None
        self._failed_at: Optional[datetime] = None  # backoff after API failure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_regime(self, news_headlines: list[str] | None = None) -> Optional[MacroRegime]:
        """Return cached regime (4h TTL). Refresh if stale. Backs off 10 min after failure."""
        now = datetime.now(timezone.utc)
        if (self._cache is not None
                and (now - self._cache.fetched_at) < timedelta(hours=self._refresh_hours)):
            return self._cache
        # Back off 10 minutes after a failed attempt to avoid per-symbol retry storms
        if self._failed_at is not None and (now - self._failed_at) < timedelta(minutes=10):
            return self._cache

        try:
            indicators = self._fetch_indicators()
            regime = self._classify(indicators, news_headlines or [])
            if regime:
                self._cache = regime
                self._failed_at = None
            else:
                self._failed_at = now
        except Exception as exc:
            logger.warning("MacroRegimeAnalyzer failed: %s", exc)
            self._failed_at = now
            return self._cache  # return stale cache on failure

        return self._cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_indicators(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for key, series_id in FRED_SERIES.items():
            val = self._fetch_fred(series_id)
            values[key] = f"{val:.2f}" if val is not None else "N/A"
        return values

    def _fetch_fred(self, series_id: str) -> Optional[float]:
        """Fetch latest observation for a FRED series. Falls back to yfinance if no API key."""
        if self._fred_api_key:
            try:
                params = urllib.parse.urlencode({
                    "series_id": series_id,
                    "api_key": self._fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": "5",
                })
                url = f"{self._fred_base_url}/series/observations?{params}"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    import json
                    data = json.loads(resp.read())
                    for obs in data.get("observations", []):
                        val_str = obs.get("value", ".")
                        if val_str and val_str != ".":
                            return float(val_str)
            except Exception as exc:
                logger.debug("FRED fetch failed for %s: %s", series_id, exc)

        # Fallback: yfinance for VIX only
        if series_id == "VIXCLS":
            try:
                import yfinance as yf
                ticker = yf.Ticker("^VIX")
                hist = ticker.history(period="2d", interval="1d")
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception as exc:
                logger.debug("yfinance VIX fallback failed: %s", exc)

        return None

    def _classify(self, indicators: dict[str, str], headlines: list[str]) -> Optional[MacroRegime]:
        headlines_str = "\n".join(f"  - {h}" for h in headlines[:20]) or "  (none available)"
        prompt = MACRO_USER_TEMPLATE.format(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            vix=indicators.get("vix", "N/A"),
            dgs10=indicators.get("dgs10", "N/A"),
            dxy=indicators.get("dxy", "N/A"),
            headlines=headlines_str,
        )
        try:
            response = self._client.complete(
                backend=self._backend,
                system_prompt=MACRO_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=512,
                temperature=0.1,
            )
            data = response.parse_json()
            return MacroRegime(
                name=str(data.get("name", "range_bound")),
                confidence=float(data.get("confidence", 0.5)),
                weight_overrides={k: float(v) for k, v in data.get("weight_overrides", {}).items()},
                rationale=str(data.get("rationale", "")),
            )
        except Exception as exc:
            logger.error("MacroRegime classification failed: %s", exc)
            return None

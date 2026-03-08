# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
LLM Strategy Orchestrator — synthesises regime, market context, strategy signals,
and recent P&L into a final buy/sell/hold decision.

The orchestrator is gated by `llm_orchestrator.enabled` in config. On any LLM
failure it returns ("hold", 1.0, None) so the caller falls back to _combine_signals().
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert operator on the equity and crypto markets. "
    "Your goal is to maximize the profit in the shortest time possible, through buy/hold/sell decisions. "
    "To operate, you have access to current portfolios, data and financial indicators, news casts, "
    "and buy/hold/sell recommendations from other strategies. "
    "You operate on individual stocks or crypto. "
    "Respond ONLY with a valid JSON object — no markdown, no explanation outside JSON."
)


class LLMStrategyOrchestrator:
    """
    Makes the final trade decision by calling an LLM with a condensed prompt
    that includes regime, indicators, alt-data, strategy signals, and recent P&L.

    Falls back to ("hold", 1.0, None) on any failure so the caller can use
    the regular _combine_signals() path.
    """

    def __init__(self, client: "LLMClient", cfg: dict) -> None:
        self._client = client
        self._cfg = cfg
        self._provider: str = str(cfg.get("provider", "gemini"))
        self._model: str | None = cfg.get("model") or None
        self._max_tokens: int = int(cfg.get("max_tokens", 1024))
        self._min_signal_score: float = float(cfg.get("min_signal_score", 0.03))
        _history_size: int = int(cfg.get("pnl_history_size", 20))
        # Open positions: symbol -> {broker, strategy, price, ts}
        self._open_entries: dict[str, dict] = {}
        # Rolling window of completed trades
        self._completed_trades: deque[dict] = deque(maxlen=_history_size)
        logger.info(
            "LLM strategy orchestrator enabled (%s / %s)",
            self._provider,
            self._model or "default",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        symbol: str,
        market_state: dict,
        signals: list[dict],
        weights: dict[str, float] | None,
    ) -> tuple[str, float, str | None]:
        """
        Returns (action, reduce_pct, strategy_name).

        Returns ("hold", 1.0, None) when:
        - No signal exceeds min_signal_score (cost guard)
        - LLM call fails for any reason
        """
        # Cost guard: skip if no strategy has meaningful confidence
        max_conf = max(
            (float(s.get("confidence", 0.0)) for s in signals if s.get("action") in ("buy", "sell")),
            default=0.0,
        )
        if max_conf < self._min_signal_score:
            return ("hold", 1.0, None)

        try:
            prompt = self._build_prompt(symbol, market_state, signals, weights)
            response = self._client.complete(
                backend=self._provider,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=0.1,
                model=self._model,
            )
            parsed = response.parse_json()
            action = str(parsed.get("action", "hold")).lower()
            if action not in ("buy", "sell", "hold"):
                action = "hold"
            reduce_pct = float(parsed.get("reduce_pct", 1.0))
            reduce_pct = max(0.0, min(1.0, reduce_pct))
            reasoning = parsed.get("reasoning", "")
            # Derive strategy_name from highest-confidence signal matching the action
            strategy_name: str | None = None
            best_conf = -1.0
            for sig in signals:
                if sig.get("action") == action:
                    c = float(sig.get("confidence", 0.0))
                    if c > best_conf:
                        best_conf = c
                        strategy_name = sig.get("name")
            if action == "hold":
                # Explicit LLM hold — return None so caller falls back
                return ("hold", 1.0, None)
            logger.info(
                "LLM orch → %s %s (reduce=%.2f, strat=%s) | %s",
                action.upper(), symbol, reduce_pct, strategy_name, reasoning,
            )
            return (action, reduce_pct, strategy_name)
        except Exception as exc:
            logger.warning("LLM orchestrator failed for %s: %s", symbol, exc)
            return ("hold", 1.0, None)

    def record_entry(self, symbol: str, broker: str, strategy: str, price: float) -> None:
        """Called when a buy fill is confirmed."""
        self._open_entries[symbol] = {
            "broker": broker,
            "strategy": strategy,
            "price": price,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def record_exit(self, symbol: str, price: float) -> None:
        """Called when a sell fill is confirmed. Computes and records trade P&L."""
        entry = self._open_entries.pop(symbol, None)
        if entry is None:
            return
        entry_price = float(entry.get("price", 0) or 0)
        if entry_price <= 0:
            return
        pnl_pct = (price - entry_price) / entry_price * 100.0
        self._completed_trades.append({
            "symbol": symbol,
            "strategy": entry.get("strategy", ""),
            "entry": entry_price,
            "exit": price,
            "pnl_pct": round(pnl_pct, 3),
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        logger.debug("LLM orch pnl recorded: %s %.2f%%", symbol, pnl_pct)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        symbol: str,
        market_state: dict,
        signals: list[dict],
        weights: dict[str, float] | None,
    ) -> str:
        is_crypto = "/" in symbol
        asset_type = "crypto" if is_crypto else "equity"

        # Regime
        regime_name = market_state.get("regime_name", "unknown")
        regime_prob = float(market_state.get("regime_probability", 0.0) or 0.0)

        # Indicators
        indicators = market_state.get("indicators", {}) or {}
        rsi = float(indicators.get("rsi", 50) or 50)
        atr_pct = float(indicators.get("atr_pct", 0.0) or 0.0)
        vwap_dev = float(indicators.get("vwap_deviation", 0.0) or 0.0)
        hurst = float(indicators.get("hurst", 0.5) or 0.5)

        # Sentiment
        llm_sent = market_state.get("llm_sentiment", {}) or {}
        sent_score = float(llm_sent.get("score", 0.5) if isinstance(llm_sent, dict) else 0.5)

        # Alt data
        alt = market_state.get("alt_data", {}) or {}
        fear_greed = float(alt.get("fear_greed_index", 50) or 50)
        oi_change = float(alt.get("oi_change_pct", 0.0) or 0.0)

        # Strategy signals
        signal_parts: list[str] = []
        for sig in signals:
            name = sig.get("name", "?")
            act = sig.get("action", "hold")
            conf = float(sig.get("confidence", 0.0))
            w = float((weights or {}).get(name, 1.0))
            signal_parts.append(f"{name} {act.upper()} {conf:.2f} (w={w:.2f})")
        signals_str = " | ".join(signal_parts) if signal_parts else "none"

        # Recent P&L
        recent = list(self._completed_trades)[-5:] if self._completed_trades else []
        if recent:
            pnls = [t["pnl_pct"] for t in recent]
            pnl_str = ", ".join(f"{p:+.1f}%" for p in pnls)
            avg_pnl = sum(pnls) / len(pnls)
            pnl_line = f"Recent P&L (last {len(pnls)}): {pnl_str}  (avg: {avg_pnl:+.2f}%)"
        else:
            pnl_line = "Recent P&L: no closed trades yet"

        prompt = (
            f"Symbol: {symbol}  Type: {asset_type}\n"
            f"Regime: {regime_name} (prob={regime_prob:.2f})\n"
            f"Indicators: RSI={rsi:.1f} ATR%={atr_pct*100:.2f}% VWAP_dev={vwap_dev*100:.2f}% Hurst={hurst:.2f}\n"
            f"Sentiment: LLM={sent_score:.2f} FearGreed={fear_greed:.0f} OI_change={oi_change:+.1f}%\n"
            f"Strategies: {signals_str}\n"
            f"{pnl_line}\n"
            'Decide: respond {"action":"buy|sell|hold","reduce_pct":0.0-1.0,"reasoning":"<20 words"}'
        )
        return prompt

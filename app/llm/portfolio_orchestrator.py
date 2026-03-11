# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""
LLM Portfolio Orchestrator — one Gemini call per cycle with the full picture.

Instead of isolated per-symbol decisions, the orchestrator sees ALL active
symbols simultaneously (signals, indicators, alt-data, regime, portfolio state,
P&L history) and returns buy/sell/hold decisions for each.  This lets the model
compare opportunities across the universe, avoid over-concentration, and make
decisions that are consistent within a cycle.

Flow:
  1. _check_execution_for_symbol() calls update_signals() for every symbol
     as signals arrive (thread-safe).
  2. After the symbol batch finishes, trader.py calls run_portfolio_cycle()
     once — this fires a single Gemini call with the full picture.
  3. Decisions are cached; get_decision(symbol) is used in the *next* cycle's
     execution path to replace _combine_signals().
  4. On any LLM failure the previous cycle's cache is kept, and per-symbol
     fallback (_combine_signals) is used for symbols with no cached decision.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a portfolio manager for a multi-asset crypto and equity trading system. "
    "You will receive, every trading cycle, a COMPLETE PICTURE of all active symbols: "
    "market regime, technical indicators (RSI, ATR, VWAP deviation, Hurst), sentiment, "
    "fear/greed index, open-interest change, insider flows, and buy/sell/hold signals "
    "with confidence scores from multiple quantitative strategies. "
    "Your goal is to maximise risk-adjusted return by deciding buy/sell/hold for EACH "
    "symbol, using the full cross-symbol context to rank opportunities and avoid "
    "over-concentration. Prefer quality over quantity: pick the best 1–5 buys per cycle "
    "rather than buying everything with a positive signal. "
    "Respond ONLY with a valid JSON object — no markdown, no explanation outside JSON."
)


class LLMPortfolioOrchestrator:
    """
    Portfolio-level LLM orchestrator.

    Collects per-symbol signal data throughout a cycle, then fires ONE Gemini
    call with all symbols visible to produce a consistent set of decisions.
    Decisions are cached for the next cycle's execution path.
    """

    def __init__(self, client: "LLMClient", cfg: dict) -> None:
        self._client = client
        self._cfg = cfg
        self._provider: str = str(cfg.get("provider", "gemini"))
        self._model: str | None = cfg.get("model") or None
        self._max_tokens: int = int(cfg.get("max_tokens", 2048))
        self._min_signal_score: float = float(cfg.get("min_signal_score", 0.03))

        _history_size: int = int(cfg.get("pnl_history_size", 20))
        self._price_invalidation_pct: float = float(cfg.get("price_invalidation_pct", 0.01))

        # Thread-safe accumulator: filled by update_signals() during each cycle
        self._signal_buffer: dict[str, dict] = {}
        self._buffer_lock = threading.Lock()

        # Decision cache: written by run_portfolio_cycle(), read by get_decision()
        self._decisions: dict[str, tuple[str, float, str | None]] = {}
        # Reference prices at time of last LLM decision (for cache invalidation)
        self._decision_prices: dict[str, float] = {}

        # P&L tracking (same semantics as LLMStrategyOrchestrator)
        self._open_entries: dict[str, dict] = {}
        self._completed_trades: deque[dict] = deque(maxlen=_history_size)

        logger.info(
            "LLM portfolio orchestrator enabled (%s / %s)",
            self._provider,
            self._model or "default",
        )

    # ------------------------------------------------------------------
    # Signal accumulation (called per-symbol, may be concurrent)
    # ------------------------------------------------------------------

    def update_signals(
        self,
        symbol: str,
        signals: list[dict],
        market_state: dict,
    ) -> None:
        """Store enriched signals for this symbol so run_portfolio_cycle() can see them."""
        with self._buffer_lock:
            self._signal_buffer[symbol] = {
                "signals": list(signals),
                "market_state": market_state,
            }

    # ------------------------------------------------------------------
    # Decision read-back (called per-symbol during execution)
    # ------------------------------------------------------------------

    def get_decision(
        self, symbol: str, current_price: float | None = None
    ) -> tuple[str, float, str | None]:
        """
        Return the portfolio-level decision for this symbol from the last cycle.

        If current_price is provided and has moved ≥ price_invalidation_pct from
        the reference price stored at decision time, the cached decision is expired
        and ("hold", 1.0, None) is returned so the caller falls back to _combine_signals().

        Returns ("hold", 1.0, None) when no cached decision exists.
        """
        cached = self._decisions.get(symbol)
        if cached is None:
            return ("hold", 1.0, None)
        if current_price and current_price > 0 and self._price_invalidation_pct > 0:
            ref = self._decision_prices.get(symbol, 0)
            if ref > 0 and abs(current_price - ref) / ref >= self._price_invalidation_pct:
                logger.debug(
                    "Decision invalidated %s: price moved %.2f%% from ref %.4f",
                    symbol,
                    abs(current_price - ref) / ref * 100,
                    ref,
                )
                return ("hold", 1.0, None)
        return cached

    # ------------------------------------------------------------------
    # Portfolio LLM call (called once after each batch)
    # ------------------------------------------------------------------

    def run_portfolio_cycle(self, portfolio_context: dict) -> None:
        """
        Fire ONE Gemini call with the full cross-symbol picture.
        Updates self._decisions for the next cycle.

        Args:
            portfolio_context: dict with keys equity, cash, crypto_exposure_pct,
                               positions {symbol: {qty, avg_entry}}.
        """
        with self._buffer_lock:
            buffer = dict(self._signal_buffer)
            self._signal_buffer.clear()

        if not buffer:
            return

        # Cost guard: skip if no symbol has a meaningful signal
        any_signal = any(
            any(
                float(s.get("confidence", 0)) >= self._min_signal_score
                and s.get("action") in ("buy", "sell")
                for s in data["signals"]
            )
            for data in buffer.values()
        )
        if not any_signal:
            logger.debug("Portfolio LLM: no signals above threshold, skipping")
            return

        try:
            prompt = self._build_portfolio_prompt(buffer, portfolio_context)
            response = self._client.complete(
                backend=self._provider,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=0.1,
                model=self._model,
            )
            parsed = response.parse_json()
            decisions_raw = parsed.get("decisions", {})

            new_decisions: dict[str, tuple[str, float, str | None]] = {}
            for sym, action_raw in decisions_raw.items():
                action = str(action_raw).lower()
                if action not in ("buy", "sell", "hold"):
                    action = "hold"
                strategy_name = self._best_strategy(
                    buffer.get(sym, {}).get("signals", []), action
                )
                new_decisions[sym] = (action, 1.0, strategy_name)

            self._decisions = new_decisions
            # Store reference prices for cache invalidation; prune stale symbols
            for sym in list(self._decision_prices):
                if sym not in new_decisions:
                    del self._decision_prices[sym]
            for sym in new_decisions:
                ms = buffer.get(sym, {}).get("market_state", {}) or {}
                lp = float(ms.get("last_price") or 0)
                if lp > 0:
                    self._decision_prices[sym] = lp
            n_buy = sum(1 for a, _, _ in new_decisions.values() if a == "buy")
            n_sell = sum(1 for a, _, _ in new_decisions.values() if a == "sell")
            logger.info(
                "Portfolio LLM → %d symbols: %d buy, %d sell, %d hold | %s",
                len(new_decisions),
                n_buy,
                n_sell,
                len(new_decisions) - n_buy - n_sell,
                str(parsed.get("reasoning", ""))[:100],
            )
        except Exception as exc:
            logger.warning("Portfolio LLM call failed: %s — keeping previous decisions", exc)

    # ------------------------------------------------------------------
    # P&L tracking (same as LLMStrategyOrchestrator)
    # ------------------------------------------------------------------

    def record_entry(self, symbol: str, broker: str, strategy: str, price: float) -> None:
        self._open_entries[symbol] = {
            "broker": broker,
            "strategy": strategy,
            "price": price,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def record_exit(self, symbol: str, price: float) -> None:
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
        logger.debug("Portfolio orch P&L: %s %.2f%%", symbol, pnl_pct)

    def get_stats(self) -> dict:
        """Return runtime stats for TacticalMetaOrchestrator metrics snapshot."""
        trades = list(self._completed_trades)
        if trades:
            wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
            win_rate = wins / len(trades)
        else:
            win_rate = 0.0
        return {
            "completed_trades": len(trades),
            "combine_win_rate": win_rate,
            # Override stats are computed from decision trace — not tracked here
            "override_rate": 0.0,
            "override_win_rate": 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_strategy(self, signals: list[dict], action: str) -> str | None:
        best_conf, best_name = -1.0, None
        for sig in signals:
            if sig.get("action") == action:
                c = float(sig.get("confidence", 0.0))
                if c > best_conf:
                    best_conf = c
                    best_name = sig.get("name")
        return best_name

    def _build_portfolio_prompt(self, buffer: dict, portfolio_ctx: dict) -> str:
        # Regime from the first available symbol's market_state
        first_ms = next(iter(buffer.values()), {}).get("market_state", {}) or {}
        regime = first_ms.get("regime_name", "unknown")
        regime_prob = float(first_ms.get("regime_probability", 0.0) or 0.0)

        # Alt-data (shared across crypto)
        alt = first_ms.get("alt_data", {}) or {}
        fear_greed = float(alt.get("fear_greed_index", 50) or 50)
        oi_change = float(alt.get("oi_change_pct", 0.0) or 0.0)

        # Portfolio
        equity = float(portfolio_ctx.get("equity", 0.0) or 0.0)
        crypto_pct = float(portfolio_ctx.get("crypto_exposure_pct", 0.0) or 0.0)
        positions = portfolio_ctx.get("positions", {}) or {}
        pos_strs = [
            f"{sym}={float(info.get('qty', 0)):.4g}"
            for sym, info in list(positions.items())[:12]
        ]
        pos_line = ", ".join(pos_strs) if pos_strs else "none"

        # Recent P&L
        recent = list(self._completed_trades)[-5:]
        if recent:
            pnls = [t["pnl_pct"] for t in recent]
            pnl_line = (
                f"Recent P&L: {', '.join(f'{p:+.1f}%' for p in pnls)}"
                f" (avg {sum(pnls)/len(pnls):+.2f}%)"
            )
        else:
            pnl_line = "Recent P&L: no closed trades yet"

        # Strategy win rates
        strat_wr = portfolio_ctx.get("strategy_win_rates") or {}
        if strat_wr:
            parts = [
                f"{s}:{v['win_rate']*100:.0f}%({v['n']})"
                for s, v in sorted(strat_wr.items(), key=lambda x: -x[1]["n"])
            ]
            strat_line = "Strategy win rates: " + " | ".join(parts) + "\n"
        else:
            strat_line = ""

        # Build per-symbol lines — compact format
        sym_lines: list[str] = []
        for sym in sorted(buffer):
            data = buffer[sym]
            ms = data.get("market_state", {}) or {}
            sigs = data.get("signals", [])

            ind = ms.get("indicators", {}) or {}
            rsi = float(ind.get("rsi", 50) or 50)
            atr_pct = float(ind.get("atr_pct", 0.0) or 0.0) * 100
            vwap_dev = float(ind.get("vwap_deviation", 0.0) or 0.0) * 100
            hurst = float(ind.get("hurst", 0.5) or 0.5)

            # Only include actionable signals in the prompt
            sig_parts = [
                f"{s.get('name', '?')[:10]} {s.get('action','?').upper()} {float(s.get('confidence',0)):.2f}"
                for s in sigs
                if s.get("action") in ("buy", "sell")
                and float(s.get("confidence", 0)) >= self._min_signal_score
            ]
            sig_str = " | ".join(sig_parts) if sig_parts else "all_hold"

            # Mark held positions
            held = positions.get(sym, {})
            held_str = f" [pos:{float(held.get('qty',0)):.4g}]" if held else ""

            # News pipeline: Ollama catalyst flag + Gemini sentiment score
            news_str = ""
            if ms.get("catalyst"):
                news_str += " CAT"
            sent = ms.get("llm_sentiment") or {}
            if sent:
                sent_bias = str(sent.get("bias", "neutral") or "neutral")[:4]
                sent_score = float(sent.get("score", 0.5) or 0.5)
                sent_rf = str(sent.get("risk_flag", "none") or "none")
                news_str += f" SENT={sent_bias}:{sent_score:.2f}"
                if sent_rf not in ("none", ""):
                    news_str += f"[{sent_rf[:4]}]"

            sym_lines.append(
                f"{sym:<14} RSI={rsi:.0f} ATR={atr_pct:.1f}% VWAP={vwap_dev:+.1f}% H={hurst:.2f}"
                f"{held_str}{news_str} | {sig_str}"
            )

        prompt = (
            f"Regime: {regime} (prob={regime_prob:.2f}) | "
            f"FearGreed={fear_greed:.0f} | OI_change={oi_change:+.1f}%\n"
            f"Portfolio equity: ${equity:.0f} | Crypto exposure: {crypto_pct:.1f}% (cap 50%)\n"
            f"Open positions: {pos_line}\n"
            f"{pnl_line}\n"
            f"{strat_line}"
            f"\n--- {len(sym_lines)} symbols (full universe) ---\n"
            + "\n".join(sym_lines)
            + "\n\n"
            "Using the complete cross-symbol picture above, decide buy/sell/hold for EACH symbol.\n"
            "Rank opportunities — prefer the 1–5 strongest setups. Avoid buying into weakness.\n"
            "Include ALL symbols in the response (even if hold).\n"
            'Respond: {"decisions":{"SYMBOL":"buy|sell|hold",...},"reasoning":"<60 words>"}'
        )
        return prompt

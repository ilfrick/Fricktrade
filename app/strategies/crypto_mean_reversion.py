# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.strategies.base import Strategy


@dataclass
class CryptoMeanReversionParams:
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 25.0
    drop_window_bars: int = 15
    hard_stop_pct: float = 3.0
    crash_filter_window: int = 100  # bars to look back for crash detection
    crash_filter_pct: float = 15.0  # suppress entries if drawdown from window high > this %


class CryptoMeanReversionStrategy(Strategy):
    """Mean reversion on crypto using Bollinger Bands + RSI.

    Buys when:
    1. Price below lower Bollinger Band (bb_std × std, bb_period bars)
    2. RSI < rsi_oversold (default 25)
    3. The drop occurred in < drop_window_bars bars (liquidation cascade signature)

    Exit: when price returns to BB midline (SMA).
    Stop: hard_stop_pct below entry.
    Only activates for crypto symbols (containing '/').
    """

    def __init__(self, params: dict):
        cfg = params.get("crypto_mean_reversion", {}) if isinstance(params, dict) else {}
        self.params = CryptoMeanReversionParams(
            bb_period=int(cfg.get("bb_period", 20)),
            bb_std=float(cfg.get("bb_std", 2.0)),
            rsi_period=int(cfg.get("rsi_period", 14)),
            rsi_oversold=float(cfg.get("rsi_oversold", 25.0)),
            drop_window_bars=int(cfg.get("drop_window_bars", 15)),
            hard_stop_pct=float(cfg.get("hard_stop_pct", 3.0)),
            crash_filter_window=int(cfg.get("crash_filter_window", 100)),
            crash_filter_pct=float(cfg.get("crash_filter_pct", 15.0)),
        )

    def generate_signal(self, market_state: dict) -> dict:
        symbol = market_state.get("symbol", "")
        if "/" not in symbol:
            return {"action": "hold", "confidence": 0.0, "name": "crypto_mean_reversion"}

        prices = market_state.get("prices", []) or []
        min_len = self.params.bb_period + self.params.rsi_period + 2
        if len(prices) < min_len:
            return {"action": "hold", "confidence": 0.0, "name": "crypto_mean_reversion"}

        close = np.array(prices, dtype=float)
        last = float(close[-1])

        # Bollinger Bands
        window = close[-self.params.bb_period :]
        sma = float(window.mean())
        std = float(window.std(ddof=1))
        lower_band = sma - self.params.bb_std * std
        upper_band = sma + self.params.bb_std * std

        # RSI
        rsi = self._compute_rsi(close, self.params.rsi_period)

        # Fast drop check — compare to price drop_window_bars ago
        ref_price = float(close[-self.params.drop_window_bars - 1]) if len(close) > self.params.drop_window_bars else float(close[0])
        fast_drop = ref_price > 0 and ((ref_price - last) / ref_price * 100.0) > 1.0

        # Check if holding a position and price has reverted to midline
        _portfolio = market_state.get("portfolio") or {}
        _pos_info = (_portfolio.get("positions") or {}).get(symbol, {})
        position_qty = float(_pos_info.get("qty", 0.0))
        if position_qty > 0 and last >= sma:
            # Only exit if meaningful profit has accumulated — prevents round-trips that
            # barely cover transaction costs when price just kisses the SMA.
            avg_entry = float(_pos_info.get("avg_entry") or 0.0)
            # If position was opened at or above SMA it was not opened by this strategy
            # (we only buy below lower_band). Don't issue sell votes for foreign positions.
            if avg_entry > 0 and avg_entry >= sma:
                return {"action": "hold", "confidence": 0.0, "name": "crypto_mean_reversion"}
            min_profit_pct = 0.5  # require at least 0.5% profit before SMA exit (must exceed ~0.40% round-trip cost)
            if avg_entry > 0:
                profit_pct = (last - avg_entry) / avg_entry * 100.0
                if profit_pct < min_profit_pct:
                    return {"action": "hold", "confidence": 0.0, "name": "crypto_mean_reversion"}
            # Scale exit confidence: higher when price is well past SMA (profit secured)
            band_range = max(upper_band - lower_band, 1e-8)
            sma_excess = (last - sma) / band_range
            sell_conf = float(min(0.5 + sma_excess * 0.5, 0.9))
            return {"action": "sell", "confidence": sell_conf, "name": "crypto_mean_reversion"}

        # Crash filter: suppress entries when price is in sustained freefall
        if self.params.crash_filter_pct > 0 and len(close) > self.params.crash_filter_window:
            window_high = float(close[-self.params.crash_filter_window:].max())
            if window_high > 0:
                drawdown_pct = (window_high - last) / window_high * 100.0
                if drawdown_pct >= self.params.crash_filter_pct:
                    return {"action": "hold", "confidence": 0.0, "name": "crypto_mean_reversion"}

        if last < lower_band and rsi < self.params.rsi_oversold and fast_drop:
            # Confidence scales with distance below lower band
            band_range = max(upper_band - lower_band, 1e-8)
            depth = (lower_band - last) / band_range
            confidence = min(0.4 + depth * 0.6, 1.0)

            # Non-price signal boosters — reward entries backed by causal data
            indicators = market_state.get("indicators") or {}
            _cascade = float(indicators.get("cascade_score", 0.0) or 0.0)
            _funding = float(indicators.get("funding_extreme", 0.0) or 0.0)
            _divergence = float(indicators.get("exchange_divergence_pct", 0.0) or 0.0)
            _social = float(indicators.get("social_velocity", 0.0) or 0.0)
            _stablecoin = float(indicators.get("stablecoin_inflow", 0.0) or 0.0)

            # Cascade: long liquidations (positive) confirm the dip is forced selling
            if _cascade > 0.3:
                confidence = min(confidence + 0.15, 1.0)
            # Funding extreme positive: longs overleveraged, unwind likely
            if _funding > 0.5:
                confidence = min(confidence + 0.10, 1.0)
            # Coinbase price higher than Binance: expect upward convergence
            if _divergence > 0.05:
                confidence = min(confidence + 0.05, 1.0)
            # Stablecoin inflow: money arriving at exchanges to buy
            if _stablecoin > 0.3:
                confidence = min(confidence + 0.05, 1.0)
            # Social velocity positive: rising attention (can be noise, small boost)
            if _social > 0.3:
                confidence = min(confidence + 0.03, 1.0)

            # Per-symbol LLM sentiment modifier (injected by trader.py via Ollama)
            llm_sent = (market_state.get("llm_sentiment") or {})
            sent_score = float(llm_sent.get("score", 0.0))
            sent_conf = float(llm_sent.get("confidence", 0.0))
            if sent_conf >= 0.3:
                # For MR: contrarian — bearish sentiment *supports* the dip-buy thesis
                # Positive score (bullish) is neutral for MR; negative (bearish) boosts
                if sent_score < -0.3:
                    confidence = min(confidence + 0.08, 1.0)
                elif sent_score > 0.5:
                    # Strong bullish during oversold? May not be a real dip — slight dampen
                    confidence *= 0.95

            return {
                "action": "buy",
                "confidence": float(confidence),
                "name": "crypto_mean_reversion",
                "hard_stop_pct": self.params.hard_stop_pct,
            }

        return {"action": "hold", "confidence": 0.0, "name": "crypto_mean_reversion"}

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        return Strategy._wilder_rsi(list(close), period)

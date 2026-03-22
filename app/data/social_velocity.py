# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick
"""Social velocity signal via CoinGecko free API.

Polls CoinGecko community data (free tier: 10-30 calls/min, no auth)
and tracks rate-of-change in social metrics (Reddit subscribers,
Twitter followers, developer activity).

Signal: social_velocity per symbol in [-1.0, 1.0]
   Positive = accelerating social attention (bullish retail flow signal)
   Negative = decelerating social attention
   0.0 = no change or no data

Polling interval: every 30 minutes (respects free tier limits).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# Map internal symbols to CoinGecko IDs
_COINGECKO_IDS = {
    "BTC/USD": "bitcoin",
    "ETH/USD": "ethereum",
    "SOL/USD": "solana",
    "DOGE/USD": "dogecoin",
    "ADA/USD": "cardano",
    "AVAX/USD": "avalanche-2",
    "LINK/USD": "chainlink",
    "DOT/USD": "polkadot",
    "UNI/USD": "uniswap",
    "XRP/USD": "ripple",
    "LTC/USD": "litecoin",
    "NEAR/USD": "near",
    "FIL/USD": "filecoin",
    "AAVE/USD": "aave",
    "ATOM/USD": "cosmos",
    "ARB/USD": "arbitrum",
    "OP/USD": "optimism",
    "RENDER/USD": "render-token",
    "SHIB/USD": "shiba-inu",
    "BONK/USD": "bonk",
}


class SocialVelocityPoller:
    """Polls CoinGecko for social metrics and computes velocity (rate of change)."""

    def __init__(self, symbols: list[str], poll_interval: int = 1800):
        self._symbols = [s for s in symbols if s in _COINGECKO_IDS]
        self._poll_interval = poll_interval
        # Store last two readings for rate of change
        self._prev_scores: dict[str, float] = {}
        self._curr_scores: dict[str, float] = {}
        self._velocity: dict[str, float] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if not _REQUESTS_AVAILABLE or not self._symbols:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="social-velocity-poller")
        self._thread.start()
        logger.info("SocialVelocityPoller started, tracking %d symbols", len(self._symbols))

    def stop(self) -> None:
        self._running = False

    def get_velocity(self, symbol: str) -> float:
        """Social velocity for symbol. 0.0 if no data."""
        with self._lock:
            return self._velocity.get(symbol, 0.0)

    def get_all_velocities(self) -> dict[str, float]:
        with self._lock:
            return dict(self._velocity)

    def _run(self) -> None:
        while self._running:
            try:
                self._poll_all()
            except Exception as exc:
                logger.warning("SocialVelocityPoller error: %s", exc)
            for _ in range(self._poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _poll_all(self) -> None:
        for symbol in self._symbols:
            if not self._running:
                break
            cg_id = _COINGECKO_IDS.get(symbol)
            if not cg_id:
                continue
            try:
                score = self._fetch_social_score(cg_id)
                if score is not None:
                    with self._lock:
                        self._prev_scores[symbol] = self._curr_scores.get(symbol, score)
                        self._curr_scores[symbol] = score
                        prev = self._prev_scores[symbol]
                        if prev > 0:
                            change_pct = (score - prev) / prev
                            # Clamp to [-1, 1], scale so 10% change = 0.5 velocity
                            self._velocity[symbol] = round(max(-1.0, min(1.0, change_pct * 5.0)), 4)
            except Exception as exc:
                logger.debug("Social fetch error %s: %s", symbol, exc)
            time.sleep(3)  # respect free tier rate limit

    def _fetch_social_score(self, cg_id: str) -> float | None:
        """Fetch a composite social score from CoinGecko community data."""
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}"
        resp = requests.get(url, params={
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        }, timeout=15)
        if resp.status_code == 429:
            logger.debug("CoinGecko rate limited, backing off")
            time.sleep(60)
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        community = data.get("community_data") or {}
        developer = data.get("developer_data") or {}
        # Composite score: Reddit + Twitter + dev activity
        reddit_subs = float(community.get("reddit_subscribers") or 0)
        reddit_active = float(community.get("reddit_accounts_active_48h") or 0)
        twitter = float(community.get("twitter_followers") or 0)
        dev_commits = float(developer.get("commit_count_4_weeks") or 0)
        # Weight: active Reddit users matter most (real-time), then Twitter, then dev
        score = reddit_active * 10.0 + reddit_subs * 0.01 + twitter * 0.001 + dev_commits * 5.0
        return score

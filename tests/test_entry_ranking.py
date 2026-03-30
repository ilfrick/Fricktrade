# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

"""Test entry ranking without importing the full TradingAgent (avoids gymnasium dep)."""

from unittest.mock import MagicMock


def _rank_entry_candidates(self, symbols):
    """Extracted ranking logic matching TradingAgent._rank_entry_candidates."""
    ranking_cfg = self.cfg.get("strategy", {}).get("entry_ranking", {})
    if not ranking_cfg.get("enabled", False):
        return symbols

    scores = []
    for sym in symbols:
        ind = self._last_indicators.get(sym) or {}
        score = 0.0

        bb_pct_b = ind.get("bollinger_pct_b")
        if bb_pct_b is not None:
            bb_pct_b = float(bb_pct_b)
            if bb_pct_b < 0.2:
                score += 0.4 * min(1.0, (0.2 - bb_pct_b) / 0.4)

        mfi_val = ind.get("mfi")
        if mfi_val is not None:
            mfi_val = float(mfi_val)
            if mfi_val < 0.3:
                score += 0.3 * min(1.0, (0.3 - mfi_val) / 0.3)

        if self._news_cache.get(sym, False):
            score += 0.15

        if self._funding_poller is not None:
            try:
                fe = self._funding_poller.get_funding_extreme(sym)
                if fe and float(fe) < 0:
                    score += 0.15
            except Exception:
                pass

        scores.append((-score, sym))

    scores.sort()
    return [sym for _, sym in scores]


def _make_agent(ranking_enabled=True):
    agent = MagicMock()
    agent.cfg = {
        "strategy": {
            "entry_ranking": {"enabled": ranking_enabled},
        }
    }
    agent._last_indicators = {}
    agent._news_cache = {}
    agent._funding_poller = None
    agent._rank_entry_candidates = lambda syms: _rank_entry_candidates(agent, syms)
    return agent


class TestEntryRanking:
    def test_ranking_disabled_returns_original_order(self):
        agent = _make_agent(ranking_enabled=False)
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        assert agent._rank_entry_candidates(symbols) == symbols

    def test_no_indicators_returns_same_symbols(self):
        agent = _make_agent()
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        assert set(agent._rank_entry_candidates(symbols)) == set(symbols)

    def test_oversold_symbol_ranked_first(self):
        agent = _make_agent()
        agent._last_indicators = {
            "BTC/USD": {"bollinger_pct_b": 0.5, "mfi": 0.5},
            "ETH/USD": {"bollinger_pct_b": -0.1, "mfi": 0.1},
            "SOL/USD": {"bollinger_pct_b": 0.3, "mfi": 0.4},
        }
        result = agent._rank_entry_candidates(["BTC/USD", "ETH/USD", "SOL/USD"])
        assert result[0] == "ETH/USD"

    def test_catalyst_boosts_ranking(self):
        agent = _make_agent()
        agent._news_cache = {"SOL/USD": True}
        agent._last_indicators = {
            "BTC/USD": {"bollinger_pct_b": 0.15, "mfi": 0.25},
            "SOL/USD": {"bollinger_pct_b": 0.15, "mfi": 0.25},
        }
        result = agent._rank_entry_candidates(["BTC/USD", "SOL/USD"])
        assert result[0] == "SOL/USD"

    def test_funding_extreme_boosts_ranking(self):
        agent = _make_agent()
        funding = MagicMock()
        funding.get_funding_extreme.side_effect = lambda sym: -1.0 if sym == "ETH/USD" else 0.0
        agent._funding_poller = funding
        agent._last_indicators = {
            "BTC/USD": {"bollinger_pct_b": 0.15, "mfi": 0.25},
            "ETH/USD": {"bollinger_pct_b": 0.15, "mfi": 0.25},
        }
        result = agent._rank_entry_candidates(["BTC/USD", "ETH/USD"])
        assert result[0] == "ETH/USD"

    def test_combined_score_ranking(self):
        agent = _make_agent()
        agent._news_cache = {"DOGE/USD": True}
        funding = MagicMock()
        funding.get_funding_extreme.side_effect = lambda sym: -1.0 if sym == "DOGE/USD" else 0.0
        agent._funding_poller = funding
        agent._last_indicators = {
            "BTC/USD": {"bollinger_pct_b": 0.5, "mfi": 0.5},
            "ETH/USD": {"bollinger_pct_b": -0.1, "mfi": 0.1},
            "DOGE/USD": {"bollinger_pct_b": -0.1, "mfi": 0.1},
        }
        result = agent._rank_entry_candidates(["BTC/USD", "ETH/USD", "DOGE/USD"])
        assert result[0] == "DOGE/USD"
        assert result[1] == "ETH/USD"
        assert result[2] == "BTC/USD"

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

import copy

from app.agents.market_state import MarketState


class TestMarketStateDefaults:
    def test_default_fields(self):
        ms = MarketState()
        assert ms.symbol == ""
        assert ms.last_price is None
        assert ms.prices == []
        assert ms.volumes == []
        assert ms.spread_pct == 0.0
        assert ms.exposure_pct == 0.0
        assert ms.regime is None
        assert ms.portfolio == {}

    def test_field_assignment(self):
        ms = MarketState(symbol="AAPL", last_price=150.0)
        assert ms.symbol == "AAPL"
        assert ms.last_price == 150.0

    def test_mutable_fields_independent(self):
        ms1 = MarketState()
        ms2 = MarketState()
        ms1.prices.append(100.0)
        assert ms2.prices == []


class TestMarketStateRoundTrip:
    def test_from_dict_to_dict(self):
        d = {
            "symbol": "TSLA",
            "last_price": 200.5,
            "prices": [198.0, 199.0, 200.5],
            "spread_pct": 0.05,
            "regime": 1,
            "regime_name": "bull",
        }
        ms = MarketState.from_dict(d)
        assert ms.symbol == "TSLA"
        assert ms.last_price == 200.5
        assert ms.prices == [198.0, 199.0, 200.5]
        assert ms.regime == 1

        result = ms.to_dict()
        assert result["symbol"] == "TSLA"
        assert result["last_price"] == 200.5
        assert result["prices"] == [198.0, 199.0, 200.5]

    def test_from_dict_ignores_unknown_keys(self):
        d = {"symbol": "AAPL", "unknown_field": 42, "another": "ignored"}
        ms = MarketState.from_dict(d)
        assert ms.symbol == "AAPL"
        assert not hasattr(ms, "unknown_field")

    def test_from_dict_empty(self):
        ms = MarketState.from_dict({})
        assert ms.symbol == ""
        assert ms.last_price is None

    def test_to_dict_complete(self):
        ms = MarketState(symbol="GOOG", exposure_pct=5.0, leverage=1.5)
        d = ms.to_dict()
        assert d["symbol"] == "GOOG"
        assert d["exposure_pct"] == 5.0
        assert d["leverage"] == 1.5
        assert "prices" in d
        assert "portfolio" in d


class TestMarketStateDeepCopy:
    def test_deepcopy_isolation(self):
        ms = MarketState(
            symbol="AAPL",
            prices=[100.0, 101.0, 102.0],
            portfolio={"equity": 100000},
        )
        ms_copy = copy.deepcopy(ms)
        ms_copy.prices.append(103.0)
        ms_copy.portfolio["cash"] = 50000

        assert len(ms.prices) == 3
        assert "cash" not in ms.portfolio

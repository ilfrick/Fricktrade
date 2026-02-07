# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from unittest.mock import MagicMock

from app.brokers.router import BrokerRouter
from app.utils.account import extract_equity_cash


def _mock_broker(account: dict, positions: list | None = None):
    broker = MagicMock()
    broker.get_account.return_value = account
    broker.get_positions.return_value = positions or []
    broker.is_connected.return_value = True
    return broker


class TestExtractEquityCash:
    def test_alpaca_format(self):
        eq, cash, bp = extract_equity_cash({"equity": "100000", "cash": "50000", "buying_power": "200000"})
        assert eq == 100000.0
        assert cash == 50000.0
        assert bp == 200000.0

    def test_ibkr_format(self):
        eq, cash, bp = extract_equity_cash({
            "NetLiquidation": 150000.0,
            "TotalCashValue": 75000.0,
            "BuyingPower": 300000.0,
        })
        assert eq == 150000.0
        assert cash == 75000.0
        assert bp == 300000.0

    def test_ibkr_available_funds_fallback(self):
        eq, cash, bp = extract_equity_cash({
            "NetLiquidation": 150000.0,
            "TotalCashValue": 75000.0,
            "AvailableFunds": 120000.0,
        })
        assert bp == 120000.0

    def test_empty_account(self):
        eq, cash, bp = extract_equity_cash({})
        assert eq == 0.0
        assert cash == 0.0
        assert bp == 0.0


class TestBrokerRouterGetAccount:
    def test_single_broker_aggregation(self):
        b1 = _mock_broker({"equity": 100000, "cash": 50000, "buying_power": 200000})
        router = BrokerRouter({"alpaca": b1})
        account = router.get_account()
        assert account["equity"] == 100000.0
        assert account["cash"] == 50000.0

    def test_multi_broker_aggregation(self):
        b1 = _mock_broker({"equity": 100000, "cash": 50000, "buying_power": 200000})
        b2 = _mock_broker({"NetLiquidation": 50000, "TotalCashValue": 25000, "BuyingPower": 100000})
        router = BrokerRouter({"alpaca": b1, "ibkr": b2})
        account = router.get_account()
        assert account["equity"] == 150000.0
        assert account["cash"] == 75000.0
        assert account["buying_power"] == 300000.0

    def test_is_connected(self):
        b1 = _mock_broker({})
        b2 = _mock_broker({})
        router = BrokerRouter({"a": b1, "b": b2})
        assert router.is_connected()
        b2.is_connected.return_value = False
        assert not router.is_connected()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations

import pytest

pytest.importorskip("ib_insync")

from app.brokers import ibkr


class _StubIB:
    def __init__(self) -> None:
        self.last_contract = None

    def connect(self, host, port, clientId):
        return None

    def isConnected(self):
        return True

    def placeOrder(self, contract, order):
        order.permId = 42
        self.last_contract = contract
        return type("Trade", (), {"order": order})()

    def sleep(self, *_args, **_kwargs):
        return None

    def positions(self):
        return []

    def trades(self):
        return []

    def accountSummary(self, **_kwargs):
        return []


def test_ibkr_currency_mapping(monkeypatch) -> None:
    stub = _StubIB()
    monkeypatch.setattr(ibkr, "IB", lambda: stub)

    broker = ibkr.IBKRBroker(
        "127.0.0.1",
        4001,
        1,
        currency="USD",
        symbol_currencies={"ENI": "EUR"},
    )

    broker.place_order("ENI", "buy", 1, "market")
    assert stub.last_contract.currency == "EUR"

    broker.place_order("AAPL", "buy", 1, "market")
    assert stub.last_contract.currency == "USD"

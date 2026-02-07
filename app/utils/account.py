# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from __future__ import annotations


def extract_equity_cash(account: dict) -> tuple[float, float, float]:
    """Extract (equity, cash, buying_power) from broker account dict.

    Handles Alpaca (equity/cash), IBKR (NetLiquidation/TotalCashValue),
    and multi-broker aggregation formats.
    """
    equity = cash = buying_power = 0.0
    if "equity" in account:
        equity = float(account.get("equity") or 0.0)
        cash = float(account.get("cash") or 0.0)
        buying_power = float(account.get("buying_power") or 0.0)
    elif "NetLiquidation" in account:
        equity = float(account.get("NetLiquidation") or 0.0)
        cash = float(account.get("TotalCashValue") or 0.0)
        buying_power = float(account.get("BuyingPower") or account.get("AvailableFunds") or 0.0)
    return equity, cash, buying_power

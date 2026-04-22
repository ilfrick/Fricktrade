# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from abc import ABC, abstractmethod


class Broker(ABC):
    @abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def close_position(self, symbol: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    def get_today_deposits(self) -> float:
        """Return net USD value of external deposits received today. Override per broker."""
        return 0.0

    def get_asset_class(self, symbol: str) -> str:
        """Return 'crypto' or 'equities' based on symbol format."""
        return "crypto" if "/" in symbol else "equities"

    def supports_short(self, symbol: str) -> bool:
        """Crypto cannot be shorted on Alpaca."""
        return self.get_asset_class(symbol) != "crypto"

    def api_account_id(self) -> str:
        """Opaque identifier for the underlying brokerage account.

        Virtual accounts sharing the same API key/credentials MUST return
        the same value so the router can detect shared positions and avoid
        double-counting.  Default: unique per instance.
        """
        return str(id(self))

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

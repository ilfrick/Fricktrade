# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Nicola Vittorio Francesconi, AKA ilfrick

from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_state: dict) -> dict:
        raise NotImplementedError

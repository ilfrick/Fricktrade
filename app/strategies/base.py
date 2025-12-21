from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_state: dict) -> dict:
        raise NotImplementedError

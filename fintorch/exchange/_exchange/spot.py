from abc import ABC, abstractmethod

from .market import Market
from .trade import Trade


class Spot(ABC):
    @property
    @abstractmethod
    def trade(self) -> Trade:
        pass

    @property
    @abstractmethod
    def market(self) -> Market:
        pass

from abc import ABC, abstractmethod

from ...dtype import Balance
from ...enum import MarketType


class Wallet(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_balance(self, market_type: MarketType, asset: str) -> Balance:
        raise NotImplementedError()

    @abstractmethod
    def transfer_balance(self, source: MarketType, destination: MarketType, asset: str, amount: float) -> bool:
        raise NotImplementedError()

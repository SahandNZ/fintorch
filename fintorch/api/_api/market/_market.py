from abc import ABC

from ._data import MarketDataEndPoints
from ._trade import MarketTradeEndPoints


class MarketEndPoints(ABC):
    def __init__(self, data: MarketDataEndPoints, trade: MarketTradeEndPoints, ) -> None:
        self.__data: MarketDataEndPoints = data
        self.__trade: MarketTradeEndPoints = trade

    @property
    def data(self) -> MarketDataEndPoints:
        return self.__data

    @property
    def trade(self) -> MarketTradeEndPoints:
        return self.__trade

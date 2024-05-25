import os
from abc import ABC

from fintorch.enum import MarketType
from fintorch.utils.directory import create_directory
from ._data import MarketData
from ._trade import MarketTrade


class Market(ABC):
    def __init__(self, data: MarketData, trade: MarketTrade, market_type: MarketType) -> None:
        self.__data: MarketData = data
        self.__trade: MarketTrade = trade
        self.__market_type: MarketType = market_type

    @property
    def data(self) -> MarketData:
        return self.__data

    @property
    def trade(self) -> MarketTrade:
        return self.__trade

    @property
    def market_type(self) -> MarketType:
        return self.__market_type

    def open(self, directory: str):
        directory = os.path.join(directory, str(self.market_type))
        self.data.open(path=os.path.join(directory, "data.pkl"))
        self.trade.open(path=os.path.join(directory, "trade.pkl"))

    def next(self, timestamp: int) -> None:
        self.data.next(timestamp=timestamp)
        self.trade.next(timestamp=timestamp)

    def close(self, directory: str):
        directory = os.path.join(directory, str(self.market_type))
        create_directory(directory)
        self.data.close(path=os.path.join(directory, "data.pkl"))
        self.trade.close(path=os.path.join(directory, "trade.pkl"))

from abc import ABC
from typing import List

from ._data import MarketData
from .trade import MarketTrade
from fintorch.enum import MarketType, TimeFrame


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

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        self.data.prepare(symbols=symbols, time_frames=time_frames)
        self.trade.prepare(symbols=symbols, time_frames=time_frames)

    def next(self, timestamp: int) -> None:
        self.data.next(timestamp=timestamp)
        self.trade.next(timestamp=timestamp)

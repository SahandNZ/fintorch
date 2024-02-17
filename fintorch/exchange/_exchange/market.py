from abc import ABC
from typing import List

from .data import Data
from .trade import Trade
from ...enum import MarketType, TimeFrame


class Market(ABC):
    def __init__(self, market_type: MarketType, data: Data, trade: Trade) -> None:
        self.__market_type: MarketType = market_type
        self.__data: Data = data
        self.__trade: Trade = trade

    @property
    def market_type(self) -> MarketType:
        return self.__market_type

    @property
    def data(self) -> Data:
        return self.__data

    @property
    def trade(self) -> Trade:
        return self.__trade

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        self.data.prepare(symbols=symbols, time_frames=time_frames)
        self.trade.prepare(symbols=symbols, time_frames=time_frames)

    def next(self, timestamp: int) -> None:
        self.data.next(timestamp=timestamp)
        self.trade.next(timestamp=timestamp)

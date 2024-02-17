from abc import ABC
from typing import List

from fintorch.enum import MarketType, TimeFrame


class Element(ABC):
    def __init__(self, exchange_name: str, market_type: MarketType, interval: TimeFrame) -> None:
        self.__exchange_name: str = exchange_name
        self.__market_type: MarketType = market_type
        self.__interval: TimeFrame = interval

        self.__timestamp: int = None
        self.__symbols: List[str] = None
        self.__time_frames: List[TimeFrame] = None

    @property
    def exchange_name(self) -> str:
        return self.__exchange_name

    @property
    def market_type(self) -> MarketType:
        return self.__market_type

    @property
    def interval(self) -> TimeFrame:
        return self.__interval

    @property
    def timestamp(self) -> int:
        return self.__timestamp

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[TimeFrame]:
        return self.__time_frames

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        self.__symbols = symbols
        self.__time_frames = time_frames

    def next(self, timestamp: int) -> None:
        self.__timestamp = timestamp

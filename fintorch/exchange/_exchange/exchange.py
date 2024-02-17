from abc import ABC
from typing import List

from .market import Market
from .wallet import Wallet
from ...enum import TimeFrame


class Exchange(ABC):
    def __init__(self, name: str, wallet: Wallet, spot: Market, future: Market):
        self.__name: str = name
        self.__wallet: Wallet = wallet
        self.__spot: Market = spot
        self.__future: Market = future

    @property
    def name(self) -> str:
        return self.__name

    @property
    def wallet(self) -> Wallet:
        return self.__wallet

    @property
    def spot(self) -> Market:
        return self.__spot

    @property
    def future(self) -> Market:
        return self.__future

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        # self.spot.prepare(symbols=symbols, time_frames=time_frames)
        self.future.prepare(symbols=symbols, time_frames=time_frames)

    def next(self, timestamp: int) -> None:
        # self.spot.next(timestamp=timestamp)
        self.future.next(timestamp=timestamp)

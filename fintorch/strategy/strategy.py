from abc import ABC, abstractmethod

import pandas as pd

from ..component import Component
from ..enum import TimeFrame
from ..exchange import Exchange


class Strategy(Component, ABC):
    def __init__(self, name: str, short_name: str, symbol: str, time_frame: TimeFrame):
        super().__init__(name=name, short_name=short_name, description="")
        self.__symbol: str = symbol
        self.__time_frame: TimeFrame = time_frame

        self.__processed_df: pd.DataFrame = None

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.__time_frame

    @property
    def processed_data(self) -> pd.DataFrame:
        return self.__processed_df

    @abstractmethod
    def prepare(self, exchange: Exchange) -> None:
        raise NotImplementedError()

    @abstractmethod
    def next(self, exchange: Exchange) -> None:
        raise NotImplementedError()

    @abstractmethod
    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

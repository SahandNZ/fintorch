from abc import ABC, abstractmethod
from typing import Union

from ..component import Component
from ..dtype import Position
from ..enum import TimeFrame
from ..exchange import Exchange, Market
from ..utils.hash import static_list_hash


class Strategy(Component, ABC):
    def __init__(self, name: str, short_name: str, symbol: str, time_frame: TimeFrame):
        super().__init__(name=name, short_name=short_name, description="")
        self.__symbol: str = symbol
        self.__time_frame: TimeFrame = time_frame

        self._static_hash: int = static_list_hash([
            self.short_name,
            self.symbol,
            int(self.time_frame * 1000)
        ])

        self.exchange: Union[Exchange, None] = None

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.__time_frame

    @property
    def static_hash(self) -> int:
        return self._static_hash

    @property
    def future(self) -> Market:
        return self.exchange.future

    @abstractmethod
    def on_new_candle(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_opened_position(self, position: Position) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_closed_position(self, position: Position) -> None:
        raise NotImplementedError()

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def __str__(self):
        return f"{self.short_name} {self.symbol} {str(self.time_frame)}"

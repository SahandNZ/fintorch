from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from .element import Element
from ...dtype import Candle, DataCollection, SymbolInfo
from ...enum import MarketType, TimeFrame


class Data(Element, ABC):
    def __init__(self, exchange_name: str, market_type: MarketType, interval: TimeFrame) -> None:
        super().__init__(exchange_name=exchange_name, market_type=market_type, interval=interval)

    @property
    @abstractmethod
    def max_candles(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_current_timestamp(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_ping(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_symbols_info(self) -> List[SymbolInfo]:
        raise NotImplementedError()

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_symbols(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_current_candle(self, symbol: str, time_frame: TimeFrame) -> Candle:
        raise NotImplementedError()

    @abstractmethod
    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def get_data_collection(self, symbols: List[str], time_frames: List[TimeFrame]) -> DataCollection:
        raise NotImplementedError()

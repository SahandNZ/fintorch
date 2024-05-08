from abc import abstractmethod, ABC
from typing import List

import pandas as pd

from ._element import MarketElement
from ....dtype import DataCollection, SymbolInfo, Ticker, Candle
from ....enum import TimeFrame, MarketType


class MarketData(MarketElement, ABC):
    def __init__(self, market_type: MarketType):
        super().__init__(market_type=market_type)

    @abstractmethod
    def get_ping(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def get_current_timestamp(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def get_symbols_info(self) -> List[SymbolInfo]:
        raise NotImplementedError()

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_symbols_ticker(self) -> List[Ticker]:
        raise NotImplementedError()

    @abstractmethod
    def get_symbol_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError()

    @abstractmethod
    def get_current_candle(self, symbol: str, time_frame: TimeFrame) -> Candle:
        raise NotImplementedError()

    @abstractmethod
    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def get_funding_rates_dataframe(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def get_top_long_short_ratios_account_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def get_data_collection(self, symbols: List[str], time_frames: List[TimeFrame]) -> DataCollection:
        raise NotImplementedError()

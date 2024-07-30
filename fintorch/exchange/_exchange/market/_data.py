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

    def get_data_collection(self, symbols: List[str], time_frames: List[TimeFrame]) -> DataCollection:
        dc = DataCollection()
        for symbol in symbols:
            symbol_info = self.get_symbol_info(symbol=symbol)
            dc.set_symbol_info(symbol=symbol, symbol_info=symbol_info)
            for time_frame in time_frames:
                df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
                dc.set_candles_df(symbol=symbol, time_frame=time_frame, df=df)

        return dc

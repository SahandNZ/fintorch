import inspect
import sys
from typing import List

import pandas as pd

from .._online import OnlineExchange
from ...dtype import Candle, DataCollection, SymbolInfo
from ...enum import MarketType, TimeFrame
from ...exchange import Data


class LocalData(Data):
    def __init__(self, market_type: MarketType, interval: TimeFrame, online_exchange: OnlineExchange) -> None:
        super().__init__(exchange_name="Local Exchange", market_type=market_type, interval=interval)
        self.__online_exchange: OnlineExchange = online_exchange
        self.__online_data: Data = getattr(self.__online_exchange, str(self.market_type)).data

    @property
    def max_candles(self) -> int:
        raise NotImplementedError()

    def get_current_timestamp(self) -> int:
        return self.timestamp

    def get_ping(self) -> int:
        return 0

    def get_symbols_info(self) -> List[SymbolInfo]:
        return self.__online_data.get_symbols_info()

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self.__online_data.get_symbol_info(symbol=symbol)

    def get_symbols(self) -> List[str]:
        return self.__online_data.get_symbols()

    def get_current_candle(self, symbol: str, time_frame: TimeFrame) -> Candle:
        df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
        lst = [df.index[-1]] + df.iloc[-1].to_list()
        candle = Candle.from_list(lst)
        return candle

    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        df = self.__online_data.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
        filtered_df = df[df.index < self.timestamp].copy()
        return filtered_df

    def get_data_collection(self, symbols: List[str], time_frames: List[TimeFrame]) -> DataCollection:
        raise NotImplementedError()

    def get_candles_whole_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)
        print('caller name:', caller_frame[1][3])
        return self.__online_data.get_candles_dataframe(symbol=symbol, time_frame=time_frame)

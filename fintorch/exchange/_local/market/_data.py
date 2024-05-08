import inspect
from typing import List

import pandas as pd

from fintorch.exchange._online import OnlineExchange
from fintorch.dtype import Candle, DataCollection, SymbolInfo
from fintorch.enum import MarketType, TimeFrame



class LocalMarketData(MarketData):
    def __init__(self, online_exchange: OnlineExchange, market_type: MarketType, interval: TimeFrame,) -> None:
        super().__init__(exchange_name="Local Exchange", market_type=market_type, interval=interval)
        self.__online_exchange: OnlineExchange = online_exchange
        self.__online_data: Data = getattr(self.__online_exchange, str(self.market_type)).data

    def get_current_timestamp(self) -> int:
        return self.timestamp

    def get_ping(self) -> float:
        return 0

    def get_symbols_info(self) -> List[SymbolInfo]:
        return self.__online_data.get_symbols_info()

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self.__online_data.get_symbol_info(symbol=symbol)

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

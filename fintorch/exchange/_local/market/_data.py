from typing import List, Dict
from datetime import datetime
from fintorch.dtype import DataCollection
import pandas as pd

from ..._exchange.market import MarketData
from ..._online import OnlineExchange, OnlineMarketData
from ....dtype import SymbolInfo, Ticker, Candle
from ....enum import MarketType, TimeFrame


class LocalMarketData(MarketData):
    def __init__(self, online_exchange: OnlineExchange, market_type: MarketType) -> None:
        super().__init__(market_type=market_type)
        self.__online_exchange: OnlineExchange = online_exchange
        self.__online_market_data: OnlineMarketData = getattr(self.__online_exchange, str(self.market_type)).data
        
        self.__candles_df_dict: Dict[str, pd.DataFrame] = {}

    def get_current_timestamp(self) -> float:
        return self.timestamp

    def get_ping(self) -> float:
        return 0

    def get_symbols_info(self) -> List[SymbolInfo]:
        return self.__online_market_data.get_symbols_info()

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return self.__online_market_data.get_symbol_info(symbol=symbol)

    def get_symbols_ticker(self) -> List[Ticker]:
        raise NotImplementedError()

    def get_symbol_ticker(self, symbol: str) -> Ticker:
        raise NotImplementedError()

    def get_current_candle(self, symbol: str, time_frame: TimeFrame) -> Candle:
        df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
        last_row = [df.index[-1]] + df.iloc[-1].to_list()
        return Candle.from_list(data=last_row)

    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        key = (symbol, time_frame)
        
        # key does not exist in candle_df_dict
        if key not in self.__candles_df_dict:
            df = self.__online_market_data.update_base_candles_df(symbol=symbol, force_update=False)
            df = self.__online_market_data.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
            self.__candles_df_dict[key] = df

        # timestamp does not exist in dataframe
        df = self.__candles_df_dict[key]
        if self.timestamp not in df.index:
            df = self.__online_market_data.update_base_candles_df(symbol=symbol, force_update=False)
            df = self.__online_market_data.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
            self.__candles_df_dict[key] = df

        df = self.__candles_df_dict[key]
        df = df[df.index < self.timestamp]

        return df
    
    def clear_state_dict(self) -> None:
        super().clear_state_dict()
        self.__candles_df_dict.clear()
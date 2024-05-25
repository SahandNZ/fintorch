from typing import List, Tuple, Dict

import pandas as pd

from ..._exchange.market import MarketData
from ..._online import OnlineExchange, OnlineMarketData
from ....dtype import DataCollection, SymbolInfo, Ticker, Candle
from ....enum import MarketType, TimeFrame


class LocalMarketData(MarketData):
    def __init__(self, online_exchange: OnlineExchange, market_type: MarketType) -> None:
        super().__init__(market_type=market_type)
        self.__online_exchange: OnlineExchange = online_exchange
        self.__online_market_data: OnlineMarketData = getattr(self.__online_exchange, str(self.market_type)).data

        self.__current_candle_dict: Dict[Tuple[str, TimeFrame], Candle] = {}
        self.__candles_dataframe_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = {}

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
        key = (symbol, time_frame)
        if key not in self.__current_candle_dict:
            df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
            last_row = [df.index[-1]] + df.iloc[-1].to_list()
            current_candle = Candle.from_list(data=last_row)

            self.__current_candle_dict[key] = current_candle

        return self.__current_candle_dict[key]

    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        key = (symbol, time_frame)
        if key not in self.__candles_dataframe_dict:
            df = self.__online_market_data.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
            df = df[df.index < self.timestamp]
            self.__candles_dataframe_dict[key] = df

        return self.__candles_dataframe_dict[key]

    def get_data_collection(self, symbols: List[str], time_frames: List[TimeFrame]) -> DataCollection:
        dc = DataCollection()
        for symbol in symbols:
            symbol_info = self.get_symbol_info(symbol=symbol)
            dc.set_symbol_info(symbol=symbol, symbol_info=symbol_info)
            for time_frame in time_frames:
                df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
                dc.set_candles_df(symbol=symbol, time_frame=time_frame, df=df)

        return dc

    def get_funding_rates_dataframe(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError()

    def get_top_long_short_ratios_account_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def next(self, timestamp: int) -> None:
        super().next(timestamp=timestamp)
        self.__current_candle_dict.clear()
        self.__candles_dataframe_dict.clear()

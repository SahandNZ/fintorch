import copy
import itertools
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd

from ._local import LocalData
from ..dtype import Candle
from ..enum import TimeFrame
from ..exchange import Exchange, Market


class LiveData:
    def __init__(self, exchange: Exchange, symbols: str, time_frames: TimeFrame):
        self.__exchange: Exchange = exchange
        self.__symbols: List[str] = symbols
        self.__time_frames: List[TimeFrame] = time_frames
        self.__pairs: List[Tuple[str, TimeFrame]] = list(itertools.product(self.symbols, self.time_frames))

        self.__local_data: LocalData = LocalData(exchange=exchange)
        self.__local_candles_dict: Dict[Tuple[str, TimeFrame], List[Candle]] = {}
        self.__live_candles_dict: Dict[Tuple[str, TimeFrame], List[Candle]] = {}

    @property
    def exchange(self) -> Exchange:
        return self.__exchange

    @property
    def market(self) -> Market:
        return self.exchange.future.market

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> TimeFrame:
        return self.__time_frames

    @property
    def pairs(self) -> List[Tuple[str, TimeFrame]]:
        return self.__pairs

    def get_candles(self, symbol: str, time_frame: TimeFrame) -> List[Candle]:
        return copy.deepcopy(self.__live_candles_dict[(symbol, time_frame)])

    def get_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        candles = self.__live_candles_dict[(symbol, time_frame)]
        df = Candle.to_dataframe(candles=candles)

        return df

    def get_dataframes_dict(self, symbols: List[str], time_frames: List[TimeFrame]) \
            -> Dict[Tuple[str, int], pd.DataFrame]:
        df_dict = {}
        for symbol, time_frame in itertools.product(symbols, time_frames):
            df_dict[(symbol, time_frame)] = self.get_dataframe(symbol=symbol, time_frame=time_frame)

        return df_dict

    def _update_local_candles(self):
        for symbol, time_frame in self.pairs:
            if (symbol, time_frame) not in self.__local_candles_dict:
                local_candle = self.__local_data.download_candles(symbol, time_frame)
                self.__local_candles_dict[(symbol, time_frame)] = local_candle
            else:
                local_candle = self.__local_candles_dict[(symbol, time_frame)]
                current_timestamp = datetime.now().timestamp()
                update_timestamp = local_candle[-1].timestamp + time_frame * int(self.market.max_candles * 0.95)
                if update_timestamp < current_timestamp:
                    local_candle = self.__local_data.download_candles(symbol, time_frame)
                self.__local_candles_dict[(symbol, time_frame)] = local_candle

    def _refine_new_candles(self, symbol: str, time_frame: TimeFrame, start_timestamp: int) -> List[Candle]:
        return self.market.get_candles(symbol, time_frame, start_timestamp)

    def _check_new_candles(self, symbol: str, time_frame: TimeFrame, candles: List[Candle], start_timestamp: int,
                           stop_timestamp: int) -> List[Candle]:
        if start_timestamp != candles[0].timestamp or stop_timestamp != candles[-1].timestamp:
            return self._refine_new_candles(symbol, time_frame, start_timestamp)
        else:
            return candles

    def _update_live_candles(self):
        for symbol, time_frame in self.pairs:
            if (symbol, time_frame) not in self.__live_candles_dict:
                local_candles = self.__local_candles_dict[(symbol, time_frame)]
                self.__live_candles_dict[(symbol, time_frame)] = local_candles

            local_candles = self.__local_candles_dict[(symbol, time_frame)]
            live_candles = self.__live_candles_dict[(symbol, time_frame)]

            last_live_candle_timestamp = live_candles[-1].timestamp
            last_local_candle_timestamp = local_candles[-1].timestamp
            stop_timestamp = datetime.now().timestamp() // time_frame * time_frame
            if last_live_candle_timestamp < stop_timestamp:
                start_timestamp = last_local_candle_timestamp + time_frame
                new_candles = self.market.get_candles(symbol, time_frame, start_timestamp)
                new_candles = self._check_new_candles(symbol, time_frame, new_candles, start_timestamp, stop_timestamp)
                live_candles = local_candles + new_candles
                self.__live_candles_dict[(symbol, time_frame)] = live_candles

    def refresh(self) -> None:
        self._update_local_candles()
        self._update_live_candles()

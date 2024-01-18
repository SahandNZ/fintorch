import os
from datetime import datetime
from typing import List

import pandas as pd
from rich.progress import Progress

from ..dtype import Candle
from ..enum import TimeFrame
from ..exchange import Exchange, Market
from ..setting import CANDLE_DIR, CANDLE_COUNTS
from ..utils.directory import create_directory


class LocalData:
    def __init__(self, exchange: Exchange):
        self.__exchange: Exchange = exchange

    @property
    def exchange(self) -> Exchange:
        return self.__exchange

    @property
    def market(self) -> Market:
        return self.exchange.future.market

    def _local_candles_path(self, symbol: str, time_frame: TimeFrame) -> str:
        exchange_symbol_directory = os.path.join(CANDLE_DIR, self.exchange.name, symbol)
        create_directory(exchange_symbol_directory)
        file_path = os.path.join(exchange_symbol_directory, f"{time_frame}.csv")

        return file_path

    def _load_candles(self, symbol: str, time_frame: TimeFrame) -> List[Candle]:
        path = self._local_candles_path(symbol=symbol, time_frame=time_frame)
        local_candles = Candle.from_csv(path) if os.path.exists(path) else []
        return local_candles

    def _refine_candles(self, symbol: str, time_frame: TimeFrame, candles: List[Candle], index: int) -> List[Candle]:
        start_timestamp = candles[index - 1].timestamp + time_frame
        stop_timestamp = candles[index].timestamp
        leftover_candles = self.market.get_candles(symbol, time_frame, start_timestamp, stop_timestamp=stop_timestamp)

        corrected_candles = candles[:index] + leftover_candles + candles[index:]
        return self._check_candles(symbol, time_frame, corrected_candles)

    def _check_candles(self, symbol: str, time_frame: TimeFrame, candles: List[Candle]) -> List[Candle]:
        for index in range(1, len(candles)):
            if candles[index - 1].timestamp + time_frame != candles[index].timestamp:
                return self._refine_candles(symbol, time_frame, candles, index)

        return candles

    def _save_candles(self, symbol: str, time_frame: TimeFrame, candles: List[Candle], mode: str = 'w+'):
        path = self._local_candles_path(symbol=symbol, time_frame=time_frame)
        Candle.to_csv(candles=candles, path=path, mode=mode)

    def download_candles(self, symbol: str, time_frame: TimeFrame, progress: Progress = None) -> List[Candle]:
        local_candles = self._load_candles(symbol=symbol, time_frame=time_frame)

        # assign value to start timestamp
        if 0 == len(local_candles):
            if CANDLE_COUNTS is None:
                symbol_info = self.market.get_symbol_info(symbol=symbol)
                start_timestamp = symbol_info.on_board_timestamp
            else:
                current_open_timestamp = datetime.now().timestamp() // time_frame * time_frame
                start_timestamp = current_open_timestamp - CANDLE_COUNTS * time_frame
        else:
            start_timestamp = local_candles[-1].timestamp + time_frame

        new_candles = self.exchange.future.market.get_candles(symbol, time_frame, start_timestamp, progress=progress)
        updated_candles = local_candles + new_candles[:-1]
        corrected_candles = self._check_candles(symbol=symbol, time_frame=time_frame, candles=updated_candles)
        self._save_candles(symbol=symbol, time_frame=time_frame, candles=corrected_candles)

        return corrected_candles

    def download_symbols_candles(self, symbols: List[str], time_frame: TimeFrame, progress: Progress = None):
        for symbol in symbols:
            self.download_candles(symbol=symbol, time_frame=time_frame, progress=progress)

    def load_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        path = self._local_candles_path(symbol=symbol, time_frame=time_frame)
        local_dataframe = Candle.load_dataframe(path)
        return local_dataframe

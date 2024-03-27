import os
import math
import pickle
import itertools
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import pandas as pd
from rich.progress import Progress

from .network import Https, Network, Wss
from .. import Data
from ...dtype import Candle, DataCollection, SymbolInfo
from ...enum import MarketType, TimeFrame
from ...setting import BASE_TIME_FRAME, CANDLES_COUNT, FINTORCH_DATA_DIR
from ...utils.directory import create_directory
from ...utils.pandas import resample_df
from ...utils.timestamp import to_timestamp


class OnlineData(Data, Network, ABC):
    def __init__(self, exchange_name: str, market_type: MarketType, interval: TimeFrame, https: Https, wss: Wss):
        Data.__init__(self, exchange_name=exchange_name, market_type=market_type, interval=interval)
        Network.__init__(self, https=https, wss=wss)

        self.__symbols_info_dict: Dict[str, SymbolInfo] = {}
        self.__candles_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = {}
        self.__last_update_timestamp: Union[int, None] = None

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        super().prepare(symbols=symbols, time_frames=time_frames)
        self.__last_update_timestamp = math.inf
        for symbol, time_frame in itertools.product(symbols, time_frames):
            df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
            self.__last_update_timestamp = min(self.__last_update_timestamp, df.index[-1])

    def next(self, timestamp: int) -> None:
        super().next(timestamp=timestamp)
        if self.__last_update_timestamp < timestamp:
            self.__last_update_timestamp = timestamp
            for symbol in self.symbols:
                self.update_candles_dataframe(symbol=symbol, time_frame=BASE_TIME_FRAME)

    def get_ping(self) -> int:
        local = datetime.now().timestamp() * 1000
        server = self.get_current_timestamp()
        ping = round(server - local)
        return ping

    def get_symbols(self) -> List[str]:
        return [symbol_info.symbol for symbol_info in self.get_symbols_info()]

    def get_symbols_info(self) -> List[SymbolInfo]:
        if 0 == len(self.__symbols_info_dict):
            self.update_symbols_info(symbol=None)

        return list(self.__symbols_info_dict.values())

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        if symbol not in self.__symbols_info_dict:
            self.update_symbols_info(symbol=symbol)

        return self.__symbols_info_dict[symbol]

    def get_current_candle(self, symbol: str, time_frame: TimeFrame) -> Candle:
        df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
        return Candle.from_list(df.iloc[-1].to_list())

    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        key = (symbol, time_frame)
        if key not in self.__candles_df_dict:
            df = self.__load_candles_dataframe(symbol=symbol, time_frame=BASE_TIME_FRAME)
            if BASE_TIME_FRAME != time_frame:
                df = resample_df(base_df=df, source_timeframe=BASE_TIME_FRAME, destination_timeframe=time_frame)
            self.__candles_df_dict[key] = df

        return self.__candles_df_dict[key]

    def get_data_collection(self, symbols: List[str], time_frames: List[TimeFrame]) -> DataCollection:
        dc = DataCollection()
        for symbol in symbols:
            symbol_info = self.get_symbol_info(symbol=symbol)
            dc.set_symbol_info(symbol=symbol, symbol_info=symbol_info)
            for time_frame in time_frames:
                df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
                dc.set_candles_df(symbol=symbol, time_frame=time_frame, df=df)

        return dc

    def update_symbols_info(self, symbol: Union[str, None]):
        self.__symbols_info_dict = self.__load_symbols_info_dict()
        if 0 == len(self.__symbols_info_dict) or (symbol is not None and symbol not in self.__symbols_info_dict):
            symbols_info_list = self._get_symbols_info()
            self.__symbols_info_dict = {symbol_info.symbol: symbol_info for symbol_info in symbols_info_list}
            self.__save_symbols_info_dict(symbols_info_dict=self.__symbols_info_dict)

    def update_candles_dataframe(self, symbol: str, time_frame: TimeFrame, progress: Progress = None):
        symbol_info = self.get_symbol_info(symbol=symbol)
        df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)

        # assign value to start timestamp
        if 0 == len(df):
            if 0 < CANDLES_COUNT:
                current_open_timestamp = datetime.now().timestamp() // int(time_frame) * int(time_frame)
                start_timestamp = current_open_timestamp - CANDLES_COUNT * time_frame
            elif symbol_info.on_board_timestamp is not None:
                start_timestamp = symbol_info.on_board_timestamp
            else:
                start_timestamp = to_timestamp(date="2019-01-01")
        else:
            start_timestamp = df.index[-1] + time_frame

        new_df = self.__send_get_candles_requests(symbol, time_frame, start_timestamp, progress=progress)
        updated_df = pd.concat([df, new_df]) if 0 != len(new_df) and 0 != len(df) else (df if 0 != len(df) else new_df)
        corrected_df = self.__check_candles_dataframe(symbol=symbol, time_frame=time_frame, df=updated_df)
        self.__save_candles_dataframe(symbol=symbol, time_frame=time_frame, df=corrected_df[:-1])
        self.__candles_df_dict[(symbol, time_frame)] = corrected_df

    def __exchange_data_directory(self) -> str:
        data_directory = str(os.path.join(FINTORCH_DATA_DIR, self.exchange_name, str(self.market_type)))
        create_directory(data_directory)
        return data_directory

    def __symbols_info_path(self):
        exchange_data_directory = self.__exchange_data_directory()
        file_path = str(os.path.join(exchange_data_directory, "symbols-info.pkl"))

        return file_path

    def __load_symbols_info_dict(self) -> Dict[str, SymbolInfo]:
        try:
            with open(self.__symbols_info_path(), "rb") as file:
                symbols_info_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            symbols_info_dict = {}

        return symbols_info_dict

    def __save_symbols_info_dict(self, symbols_info_dict: Dict[str, SymbolInfo]) -> None:
        with open(self.__symbols_info_path(), "wb+") as file:
            pickle.dump(symbols_info_dict, file)

    def __candles_dataframe_path(self, symbol: str, time_frame: TimeFrame) -> str:
        exchange_data_directory = self.__exchange_data_directory()
        candles_directory = str(os.path.join(exchange_data_directory, "candles", symbol))
        file_path = str(os.path.join(candles_directory, f"{time_frame}.csv"))
        create_directory(candles_directory)

        return file_path

    def __load_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        path = self.__candles_dataframe_path(symbol=symbol, time_frame=time_frame)
        if os.path.exists(path):
            return Candle.load_dataframe(path=path)
        else:
            return Candle.to_dataframe(candles=[])

    def __refine_candles_dataframe(self, symbol: str, time_frame: TimeFrame, df: pd.DataFrame) -> pd.DataFrame:
        indices = df.index.to_series()
        diff_indices = indices.diff()
        missed_values = time_frame != diff_indices
        for index, value in enumerate(missed_values):
            if value:
                raise NotImplementedError()

    def __check_candles_dataframe(self, symbol: str, time_frame: TimeFrame, df: pd.DataFrame) -> pd.DataFrame:
        indices = df.index.to_series()
        diff_indices = indices.diff().dropna()
        missed_values = time_frame != diff_indices
        if missed_values.max():
            self.__refine_candles_dataframe(symbol=symbol, time_frame=time_frame, df=df)

        return df

    def __save_candles_dataframe(self, symbol: str, time_frame: TimeFrame, df: pd.DataFrame) -> None:
        path = self.__candles_dataframe_path(symbol=symbol, time_frame=time_frame)
        df.to_csv(path_or_buf=path)

    def __send_get_candles_requests(self, symbol: str, time_frame: TimeFrame, start_timestamp: int,
                                    stop_timestamp: int = None, progress: Union[Progress, bool] = None) -> pd.DataFrame:
        if stop_timestamp is None:
            stop_timestamp = datetime.now().timestamp() // int(time_frame) * int(time_frame) + int(time_frame)
        candles_count = (stop_timestamp - start_timestamp) // int(time_frame)
        requests_count = math.ceil(candles_count / self.max_candles)

        items = list(range(requests_count))
        if progress is not None:
            description = f"Downloading {symbol} {time_frame} Candles"
            task = progress.add_task(description=description, total=len(items))
        else:
            task = None

        candles: List[Candle] = []
        for index in items:
            # assign value to req_start_ts as request start timestamp
            if 0 == len(candles):
                req_start_ts = start_timestamp + index * self.max_candles * time_frame
            else:
                req_start_ts = candles[-1].timestamp + time_frame

            # assign value to req_stop_ts as request stop timestamp
            req_stop_ts = req_start_ts + self.max_candles * time_frame
            if stop_timestamp < req_stop_ts:
                req_stop_ts = stop_timestamp

            # send request
            if req_start_ts < req_stop_ts:
                req_candles = self._get_historical_candles(
                    symbol=symbol,
                    time_frame=time_frame,
                    start_timestamp=req_start_ts,
                    stop_timestamp=req_stop_ts
                )
                candles.extend(req_candles)

            # update progress bar
            if progress is not None:
                progress.update(task, advance=1)

        # make progress bar invisible
        if progress is not None:
            progress.update(task, visible=False)

        return Candle.to_dataframe(candles)

    @abstractmethod
    def _get_symbols_info(self) -> List[SymbolInfo]:
        raise NotImplementedError()

    @abstractmethod
    def _get_recent_candles(self, symbol: str, time_frame: TimeFrame) -> List[Candle]:
        raise NotImplementedError()

    @abstractmethod
    def _get_historical_candles(self, symbol: str, time_frame: TimeFrame, start_timestamp: int, stop_timestamp: int) \
            -> List[Candle]:
        raise NotImplementedError()

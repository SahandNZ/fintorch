import os
import time
import pickle
import filelock
from datetime import datetime
from typing import Dict, List

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from rich.progress import Progress

from ._error import *
from ..._exchange.market import MarketData
from ....api import API, MarketDataEndPoints
from ....dtype import Candle, SymbolInfo, Ticker
from ....enum import TimeFrame, MarketType
from ....settings import FINTORCH_DATA_DIR, BASE_TIME_FRAME
from ....utils.directory import create_directory
from ....utils.resample import resample_candles_df
from ....utils.timestamp import to_timestamp, floor_timestamp


class OnlineMarketData(MarketData):
    def __init__(self, api: API, market_type: MarketType):
        super().__init__(market_type=market_type)
        self.__api: API = api
        self.__market_data_end_points: MarketDataEndPoints = getattr(api, str(market_type)).data

        self.__data_directory: str = str(os.path.join(FINTORCH_DATA_DIR, self.api.name, str(self.market_type)))
        create_directory(self.__data_directory)

    @property
    def api(self) -> API:
        return self.__api

    @property
    def market_data_end_points(self) -> MarketDataEndPoints:
        return self.__market_data_end_points

    def get_current_timestamp(self) -> float:
        return self.market_data_end_points.get_server_timestamp()

    def get_ping(self) -> float:
        return self.market_data_end_points.get_ping()

    @property
    def __symbols_info_file_path(self) -> str:
        return str(os.path.join(self.__data_directory, "symbols-info.pkl"))

    def __load_symbols_info(self) -> Dict[str, SymbolInfo]:
        try:
            with open(self.__symbols_info_file_path, "rb") as file:
                symbols_info_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            symbols_info_dict = {}

        return symbols_info_dict

    def __save_symbols_info(self, symbols_info_dict: Dict[str, SymbolInfo]) -> None:
        with open(self.__symbols_info_file_path, "wb+") as file:
            pickle.dump(symbols_info_dict, file)

    def __update_and_save_symbols_info(self) -> Dict[str, SymbolInfo]:
        symbols_info_list = self.market_data_end_points.get_symbols_info()
        symbols_info_dict = {symbol_info.symbol: symbol_info for symbol_info in symbols_info_list}
        self.__save_symbols_info(symbols_info_dict=symbols_info_dict)

        return symbols_info_dict

    def get_symbols_info(self) -> List[SymbolInfo]:
        symbols_info_dict = self.__load_symbols_info()
        if 0 == len(symbols_info_dict):
            symbols_info_dict = self.__update_and_save_symbols_info()

        return list(symbols_info_dict.values())

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        symbols_info_dict = self.__load_symbols_info()
        if 0 == len(symbols_info_dict) or symbol not in symbols_info_dict:
            symbols_info_dict = self.__update_and_save_symbols_info()

        return symbols_info_dict[symbol]

    def get_symbols_ticker(self) -> List[Ticker]:
        return self.market_data_end_points.get_symbols_ticker()

    def get_symbol_ticker(self, symbol: str) -> Ticker:
        return self.market_data_end_points.get_symbol_ticker(symbol=symbol)

    def __get_candles_file_path(self, symbol: str) -> str:
        directory = str(os.path.join(self.__data_directory, "candles"))
        path = str(os.path.join(directory, f"{symbol}.csv"))
        create_directory(directory)

        return path

    def __get_candles_lock_path(self, symbol: str) -> str:
        return self.__get_candles_file_path(symbol=symbol) + ".lock" 
    
    def __load_candles_dataframe(self, symbol: str) -> pd.DataFrame:
        try:
            path = self.__get_candles_file_path(symbol=symbol)
            return Candle.load_dataframe(path=path)
        except:
            return Candle.to_dataframe(candles=[])

    def __save_candles_dataframe(self, symbol: str, df: pd.DataFrame) -> None:
        path = self.__get_candles_file_path(symbol=symbol)
        df.to_csv(path_or_buf=path)
            
    def __check_candles_dataframe(self, df: pd.DataFrame, check_last_candle: bool) -> pd.DataFrame:
        timestamps = df.index.to_series()
        
        # check length of dataframe
        if 0 == len(timestamps):
            raise EmptyDataFrameException("Candles dataframe is empty.")
        
        # check for missing candles
        diff_timestamps = timestamps.diff().dropna()
        is_there_missed_candles = (BASE_TIME_FRAME != diff_timestamps).any()
        if is_there_missed_candles:
            raise MissingCandlesException()
        
        # check timestamps to be divisible by base time frame
        is_there_non_divisible_candles = (0 != timestamps % BASE_TIME_FRAME).any()
        if is_there_non_divisible_candles:
            raise WrongCandleException("Wrong candles in dataframe.")
        
        # check for redundant candles
        is_there_duplicated_candles = timestamps.duplicated().any()
        if is_there_duplicated_candles:
            raise DuplicatedCandlesException("Duplicated candles in dataframe.")
        
        # check for existence of last candle
        if check_last_candle:
            current_timestamp = int(datetime.now().timestamp())
            last_open_timestamp = floor_timestamp(timestamp=current_timestamp, time_frame=BASE_TIME_FRAME)
            is_last_candle_missed = timestamps.iloc[-1] < last_open_timestamp
            if is_last_candle_missed:
                raise MissingLastCandleException("Last candle was missed in dataframe.")

        return df

    def update_base_candles_df(
        self,
        symbol: str,
        force_update: bool=False,
        progress: Progress = None
    ) -> None:
        lock_path = self.__get_candles_lock_path(symbol=symbol)
        lock = filelock.FileLock(lock_path)

        try:
            lock.acquire()
            
            # load candles df
            symbol_info = self.get_symbol_info(symbol=symbol)
            df = self.__load_candles_dataframe(symbol=symbol)

            # calculate start timestamp
            start_timestamp = df.index[-2] if 2 < len(df) else symbol_info.on_board_timestamp

            current_timestamp = datetime.now().timestamp()
            current_open_timestamp = floor_timestamp(current_timestamp, time_frame=BASE_TIME_FRAME)
            need_update = 0 == len(df) or df.index[-1] < current_open_timestamp
            if force_update or need_update: 
                # get candles from api
                new_candles = self.api.future.data.get_candles(
                    symbol=symbol,
                    time_frame=BASE_TIME_FRAME,
                    start_timestamp=start_timestamp,
                    stop_timestamp=None,
                    progress=progress
                )

                # cancat dataframes and set timestamps as index
                old_df = df.reset_index()
                new_df = Candle.to_dataframe(new_candles).reset_index()
                if 0 < len(old_df) and 0 < len(new_df):
                    df = pd.concat([old_df, new_df]).drop_duplicates(subset=["timestamp"], keep="last")
                elif 0 < len(new_df):
                    df = new_df
                else:
                    df = old_df
                df.set_index("timestamp", inplace=True)

                # check correctness of candles and save
                df = self.__check_candles_dataframe(df=df, check_last_candle=False)
                self.__save_candles_dataframe(symbol=symbol, df=df)
        except Exception as e:
            lock.release()
            raise e

    @retry(
        reraise=True,
        retry=retry_if_exception_type(MissingLastCandleException),
        stop=stop_after_attempt(5),
        wait=wait_fixed(10)
    )
    def get_base_candles_df(self, symbol: str) -> pd.DataFrame:
        # wait for writer until its work to be done
        lock_path = self.__get_candles_lock_path(symbol=symbol)
        lock = filelock.FileLock(lock_path)
        while lock.is_locked:
            time.sleep(0.1)

        # load candles df
        df = self.__load_candles_dataframe(symbol=symbol)
        self.__check_candles_dataframe(df=df, check_last_candle=True)

        return df

    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        df = self.get_base_candles_df(symbol=symbol)
        if BASE_TIME_FRAME != time_frame:
            df = resample_candles_df(df=df, source_timeframe=BASE_TIME_FRAME, destination_timeframe=time_frame)
            
        return df
    
    def get_current_candle(self, symbol: str, time_frame: TimeFrame) -> Candle:
        raise NotImplementedError()

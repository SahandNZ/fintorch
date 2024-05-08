import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Union, Type

import pandas as pd
from rich.progress import Progress

from ..._exchange.market import MarketData
from ....api import API, MarketDataEndPoints
from ....dtype import Candle, DataCollection, SymbolInfo, FundingRate, LongShortRatio, Ticker
from ....enum import TimeFrame, MarketType
from ....setting import BASE_TIME_FRAME, FINTORCH_DATA_DIR
from ....utils.directory import create_directory
from ....utils.resample import resample_candles_df, resample_long_short_ratio_df
from ....utils.timestamp import to_timestamp, round_timestamp


class OnlineMarketData(MarketData):
    def __init__(self, api: API, market_type: MarketType):
        super().__init__()
        self.__api: API = api
        self.__market_type: MarketType = market_type
        self.__market_data_end_points: MarketDataEndPoints = getattr(getattr(api, str(market_type)), "data")

        self.__update_timestamp: Union[int, None] = None
        self.__symbol_to_symbol_info: Dict[str, SymbolInfo] = {}
        self.__stf_to_candles_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = {}
        self.__stf_to_funding_rates_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = {}
        self.__stf_to_top_long_short_ratios_account_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = {}

        self.__data_directory: str = str(os.path.join(FINTORCH_DATA_DIR, self.api.name, str(self.market_type)))
        create_directory(self.__data_directory)

    @property
    def api(self) -> API:
        return self.__api

    @property
    def market_type(self) -> MarketType:
        return self.__market_type

    @property
    def market_data_end_points(self) -> MarketDataEndPoints:
        return self.__market_data_end_points

    def prepare(self, symbols: List[str], time_frames: List[TimeFrame]) -> None:
        super().prepare(symbols=symbols, time_frames=time_frames)

    def next(self, timestamp: int) -> None:
        super().next(timestamp=timestamp)
        if self.__update_timestamp is None or self.__update_timestamp < timestamp:
            self.__update_timestamp = timestamp
            for symbol in self.symbols:
                self.update_candles_dataframe(symbol=symbol)
                self.update_funding_rates_dataframe(symbol=symbol)

    def get_current_timestamp(self) -> float:
        return self.market_data_end_points.get_server_timestamp()

    def get_ping(self) -> float:
        return self.market_data_end_points.get_ping()

    def get_symbols_info(self) -> List[SymbolInfo]:
        if 0 == len(self.__symbol_to_symbol_info):
            self.__load_symbol_to_symbol_info()
        if 0 == len(self.__symbol_to_symbol_info):
            self.__update_symbols_info()

        return list(self.__symbol_to_symbol_info.values())

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        if 0 == len(self.__symbol_to_symbol_info):
            self.__load_symbol_to_symbol_info()
        if symbol not in self.__symbol_to_symbol_info:
            self.__update_symbols_info()

        return self.__symbol_to_symbol_info.get(symbol, None)

    @property
    def __symbols_info_path(self) -> str:
        return str(os.path.join(self.__data_directory, "symbols-info.pkl"))

    def __load_symbol_to_symbol_info(self) -> None:
        try:
            with open(self.__symbols_info_path, "rb") as file:
                self.__symbol_to_symbol_info = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.__symbol_to_symbol_info = {}

    def __update_symbols_info(self) -> None:
        symbols_info_list = self.market_data_end_points.get_symbols_info()
        self.__symbol_to_symbol_info = {symbol_info.symbol: symbol_info for symbol_info in symbols_info_list}
        self.__save_symbol_to_symbol_info()

    def __save_symbol_to_symbol_info(self) -> None:
        with open(self.__symbols_info_path, "wb+") as file:
            pickle.dump(self.__symbol_to_symbol_info, file)

    def get_symbols_ticker(self) -> List[Ticker]:
        return self.market_data_end_points.get_symbols_ticker()

    def get_symbol_ticker(self, symbol: str) -> Ticker:
        return self.market_data_end_points.get_symbol_ticker(symbol=symbol)

    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        key = (symbol, BASE_TIME_FRAME)
        if key not in self.__stf_to_candles_df_dict:
            df = self.__load_time_series_dataframe(
                symbol=symbol,
                time_frame=BASE_TIME_FRAME,
                attribute_name="candles",
                dtype=Candle
            )
            self.__stf_to_candles_df_dict[key] = df

        df = self.__stf_to_candles_df_dict[key]
        if BASE_TIME_FRAME != time_frame:
            df = resample_candles_df(df=df, source_timeframe=BASE_TIME_FRAME, destination_timeframe=time_frame)

        return df

    def update_candles_dataframe(self, symbol: str, progress: Progress = None) -> None:
        self.__update_time_series_dataframe(
            symbol=symbol,
            time_frame=BASE_TIME_FRAME,
            attribute_name="candles",
            dtype=Candle,
            progress=progress
        )

    def get_funding_rates_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        key = (symbol, self.market_data_end_points.funding_rates_interval)
        if key not in self.__stf_to_funding_rates_df_dict:
            df = self.__load_time_series_dataframe(
                symbol=symbol,
                time_frame=self.market_data_end_points.funding_rates_interval,
                attribute_name="funding_rates",
                dtype=FundingRate,
            )
            self.__stf_to_funding_rates_df_dict[key] = df

        return self.__stf_to_funding_rates_df_dict[key]

    def update_funding_rates_dataframe(self, symbol: str, progress: Progress = None) -> None:
        self.__update_time_series_dataframe(
            symbol=symbol,
            time_frame=self.market_data_end_points.funding_rates_interval,
            attribute_name="funding_rates",
            dtype=FundingRate,
            progress=progress
        )

    def get_top_long_short_ratios_account_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        key = (symbol, BASE_TIME_FRAME)
        if key not in self.__stf_to_top_long_short_ratios_account_df_dict:
            df = self.__load_time_series_dataframe(
                symbol=symbol,
                time_frame=BASE_TIME_FRAME,
                attribute_name="top_long_short_ratios_account",
                dtype=LongShortRatio
            )
            self.__stf_to_top_long_short_ratios_account_df_dict[key] = df

        df = self.__stf_to_top_long_short_ratios_account_df_dict[key]
        if BASE_TIME_FRAME != time_frame:
            df = resample_long_short_ratio_df(df=df, source_timeframe=BASE_TIME_FRAME, destination_timeframe=time_frame)

        return df

    def update_top_long_short_ratios_account_dataframe(self, symbol: str, progress: Progress = None) -> None:
        self.__update_time_series_dataframe(
            symbol=symbol,
            time_frame=BASE_TIME_FRAME,
            attribute_name="top_long_short_ratios_account",
            dtype=LongShortRatio,
            progress=progress
        )

    def get_data_collection(self, symbols: List[str], time_frames: List[TimeFrame]) -> DataCollection:
        dc = DataCollection()
        for symbol in symbols:
            symbol_info = self.get_symbol_info(symbol=symbol)
            funding_rates_df = self.get_funding_rates_dataframe(
                symbol=symbol,
                time_frame=self.market_data_end_points.funding_rates_interval
            )
            dc.set_symbol_info(symbol=symbol, symbol_info=symbol_info)
            dc.set_funding_rates_df(symbol=symbol, df=funding_rates_df)
            for time_frame in time_frames:
                df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
                dc.set_candles_df(symbol=symbol, time_frame=time_frame, df=df)

        return dc

    # region private timeseries methods
    def __time_series_path(self, symbol: str, time_frame: TimeFrame, attribute_name: str) -> str:
        root_directory_name = attribute_name.replace("_", "-")
        directory = str(os.path.join(self.__data_directory, root_directory_name, symbol))
        path = str(os.path.join(directory, f"{int(time_frame)}.csv"))
        create_directory(directory)

        return path

    def __load_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            attribute_name: str,
            dtype: Type[Union[FundingRate, Candle, LongShortRatio]]
    ) -> pd.DataFrame:
        path = self.__time_series_path(symbol=symbol, time_frame=time_frame, attribute_name=attribute_name)
        return dtype.load_dataframe(path=path) if os.path.exists(path) else dtype.to_dataframe([])

    def __update_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            attribute_name: str,
            dtype: Type[Union[FundingRate, Candle, LongShortRatio]],
            progress: Progress = None,
    ) -> None:
        get_attribute_dataframe = getattr(self, f"get_{attribute_name}_dataframe")
        attribute_df_dict = getattr(self, f"_OnlineMarketData__stf_to_{attribute_name}_df_dict")
        attribute_market_end_point_get_fn = getattr(self.market_data_end_points, f"get_{attribute_name}")
        symbol_info = self.get_symbol_info(symbol=symbol)
        df = get_attribute_dataframe(symbol=symbol, time_frame=time_frame)[:-1]

        # calculate start timestamp
        if 0 == len(df):
            start_timestamp = symbol_info.on_board_timestamp or to_timestamp(date="2019-01-01")
        else:
            start_timestamp = df.index[-1] + time_frame

        # calculate stop timestamp
        current_timestamp = int(datetime.now().timestamp())
        stop_timestamp = round_timestamp(timestamp=current_timestamp, time_frame=time_frame, side="up")

        results = attribute_market_end_point_get_fn(
            symbol=symbol,
            time_frame=time_frame,
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
            progress=progress
        )
        new_df = dtype.to_dataframe(results)
        updated_df = pd.concat([df, new_df]) if 0 != len(new_df) and 0 != len(df) else (df if 0 != len(df) else new_df)

        corrected_df = self.__check_time_series_dataframe(
            symbol=symbol,
            time_frame=time_frame,
            df=updated_df,
            attribute_name=attribute_name,
            dtype=dtype,
        )

        self.__save_time_series_dataframe(
            symbol=symbol,
            time_frame=time_frame,
            df=corrected_df[:-1],
            attribute_name=attribute_name,
        )

        attribute_df_dict[(symbol, time_frame)] = corrected_df

    def __check_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            df: pd.DataFrame,
            attribute_name: str,
            dtype: Type[Union[FundingRate, Candle, LongShortRatio]],
    ) -> pd.DataFrame:
        indices = df.index.to_series()
        diff_indices = indices.diff().dropna()
        missed_values = time_frame != diff_indices
        if missed_values.max():
            df = self.__refine_time_series_dataframe(
                symbol=symbol,
                time_frame=time_frame,
                df=df,
                attribute_name=attribute_name,
                dtype=dtype
            )

        return df

    def __refine_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            df: pd.DataFrame,
            attribute_name: str,
            dtype: Type[Union[FundingRate, Candle, LongShortRatio]],
    ) -> pd.DataFrame:
        raise NotImplementedError()

    def __save_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            df: pd.DataFrame,
            attribute_name: str
    ) -> None:
        path = self.__time_series_path(symbol=symbol, time_frame=time_frame, attribute_name=attribute_name)
        df.to_csv(path_or_buf=path)

    # endregion

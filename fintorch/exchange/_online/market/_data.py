import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Union, Type

import pandas as pd
from rich.progress import Progress

from ..._exchange.market import MarketData
from ....api import API, MarketDataEndPoints
from ....dtype import Candle, DataCollection, SymbolInfo, FundingRate, LongShortRatio, Ticker, AggregatedTrade
from ....enum import TimeFrame, MarketType
from ....settings import FINTORCH_DATA_DIR, BASE_TIME_FRAME
from ....utils.directory import create_directory
from ....utils.function import call_with_dict
from ....utils.resample import resample_candles_df, resample_long_short_ratio_df
from ....utils.timestamp import to_timestamp, floor_timestamp, ceil_timestamp


class OnlineMarketData(MarketData):
    def __init__(self, api: API, market_type: MarketType):
        super().__init__(market_type=market_type)
        self.__api: API = api
        self.__market_data_end_points: MarketDataEndPoints = getattr(api, str(market_type)).data

        self.__update_timestamp: Union[int, None] = None
        self.__symbol_info_dict: Dict[str, SymbolInfo] = {}
        self.__funding_rates_df_dict: Dict[str, pd.DataFrame] = {}
        self.__aggregated_trades_df_dict: Dict[str, pd.DataFrame] = {}
        self.__candles_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = {}
        self.__top_long_short_ratios_account_df_dict: Dict[Tuple[str, TimeFrame], pd.DataFrame] = {}

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

    def get_symbols_info(self) -> List[SymbolInfo]:
        if 0 == len(self.__symbol_info_dict):
            self.__load_symbol_to_symbol_info()
        if 0 == len(self.__symbol_info_dict):
            self.__update_symbols_info()

        return list(self.__symbol_info_dict.values())

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        if 0 == len(self.__symbol_info_dict):
            self.__load_symbol_to_symbol_info()
            self.__update_symbols_info()

        return self.__symbol_info_dict.get(symbol, None)

    @property
    def __symbols_info_path(self) -> str:
        return str(os.path.join(self.__data_directory, "symbols-info.pkl"))

    def __load_symbol_to_symbol_info(self) -> None:
        try:
            with open(self.__symbols_info_path, "rb") as file:
                self.__symbol_info_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.__symbol_info_dict = {}

    def __update_symbols_info(self) -> None:
        symbols_info_list = self.market_data_end_points.get_symbols_info()
        self.__symbol_info_dict = {symbol_info.symbol: symbol_info for symbol_info in symbols_info_list}
        self.__save_symbol_to_symbol_info()

    def __save_symbol_to_symbol_info(self) -> None:
        with open(self.__symbols_info_path, "wb+") as file:
            pickle.dump(self.__symbol_info_dict, file)

    def get_symbols_ticker(self) -> List[Ticker]:
        return self.market_data_end_points.get_symbols_ticker()

    def get_symbol_ticker(self, symbol: str) -> Ticker:
        return self.market_data_end_points.get_symbol_ticker(symbol=symbol)

    def get_aggregated_trades_dataframe(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.__aggregated_trades_df_dict:
            df = self.__load_time_series_dataframe(
                symbol=symbol,
                time_frame=self.market_data_end_points.aggregated_trades_interval,
                attribute_name="aggregated_trades",
                dtype=AggregatedTrade
            )
            self.__aggregated_trades_df_dict[symbol] = df

        return self.__aggregated_trades_df_dict[symbol]

    def update_aggregated_trades_dataframe(self, symbol: str, progress: Progress = None) -> None:
        self.__update_time_series_dataframe(
            symbol=symbol,
            time_frame=self.market_data_end_points.aggregated_trades_interval,
            attribute_name="aggregated_trades",
            dtype=AggregatedTrade,
            check_for_missing_values=False,
            progress=progress
        )

    def get_current_candle(self, symbol: str, time_frame: TimeFrame) -> Candle:
        df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
        last_row = [df.index[-1]] + df.iloc[-1].to_list()
        return Candle.from_list(data=last_row)

    def get_candles_dataframe(self, symbol: str, time_frame: TimeFrame) -> pd.DataFrame:
        key = (symbol, BASE_TIME_FRAME)
        if key not in self.__candles_df_dict:
            df = self.__load_time_series_dataframe(
                symbol=symbol,
                time_frame=BASE_TIME_FRAME,
                attribute_name="candles",
                dtype=Candle
            )
            self.__candles_df_dict[key] = df

        df = self.__candles_df_dict[key]
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

    def get_funding_rates_dataframe(self, symbol: str) -> pd.DataFrame:
        if symbol not in self.__funding_rates_df_dict:
            df = self.__load_time_series_dataframe(
                symbol=symbol,
                time_frame=self.market_data_end_points.funding_rates_interval,
                attribute_name="funding_rates",
                dtype=FundingRate,
            )
            self.__funding_rates_df_dict[symbol] = df

        return self.__funding_rates_df_dict[symbol]

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
        if key not in self.__top_long_short_ratios_account_df_dict:
            df = self.__load_time_series_dataframe(
                symbol=symbol,
                time_frame=BASE_TIME_FRAME,
                attribute_name="top_long_short_ratios_account",
                dtype=LongShortRatio
            )
            self.__top_long_short_ratios_account_df_dict[key] = df

        df = self.__top_long_short_ratios_account_df_dict[key]
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
            funding_rates_df = self.get_funding_rates_dataframe(symbol=symbol)
            dc.set_symbol_info(symbol=symbol, symbol_info=symbol_info)
            dc.set_funding_rates_df(symbol=symbol, df=funding_rates_df)
            for time_frame in time_frames:
                df = self.get_candles_dataframe(symbol=symbol, time_frame=time_frame)
                dc.set_candles_df(symbol=symbol, time_frame=time_frame, df=df)

        return dc

    def next(self, timestamp: int, force_update: bool = False) -> None:
        super().next(timestamp=timestamp)

        # update candles
        candles_open_timestamp = floor_timestamp(timestamp=timestamp, time_frame=BASE_TIME_FRAME)
        for (symbol, _) in self.__candles_df_dict.keys():
            df = self.get_candles_dataframe(symbol=symbol, time_frame=BASE_TIME_FRAME)
            if force_update or candles_open_timestamp not in df.index:
                self.update_candles_dataframe(symbol=symbol)

        # update funding rates
        # funding_rate_time_frame = self.market_data_end_points.funding_rates_interval
        # funding_rate_open_timestamp = floor_timestamp(timestamp=timestamp, time_frame=funding_rate_time_frame)
        # for symbol in self.__funding_rates_df_dict.keys():
        #     df = self.get_funding_rates_dataframe(symbol=symbol)
        #     if force_update or funding_rate_open_timestamp not in df.index:
        #         self.update_funding_rates_dataframe(symbol=symbol)

    # region private timeseries methods
    def __time_series_path(self, symbol: str, time_frame: TimeFrame, attribute_name: str) -> str:
        root_directory_name = attribute_name.replace("_", "-")
        directory = str(os.path.join(self.__data_directory, root_directory_name, symbol))
        path = str(os.path.join(directory, f"{str(time_frame)}.csv"))
        create_directory(directory)

        return path

    def __load_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            attribute_name: str,
            dtype: Type[Union[AggregatedTrade, Candle, FundingRate, LongShortRatio]]
    ) -> pd.DataFrame:
        path = self.__time_series_path(symbol=symbol, time_frame=time_frame, attribute_name=attribute_name)
        return dtype.load_dataframe(path=path) if os.path.exists(path) else dtype.to_dataframe([])

    def __update_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            attribute_name: str,
            dtype: Type[Union[AggregatedTrade, Candle, FundingRate, LongShortRatio]],
            check_for_missing_values: bool = True,
            progress: Progress = None,
    ) -> None:
        attribute_get_dataframe = getattr(self, f"get_{attribute_name}_dataframe")
        attribute_df_dict = getattr(self, f"_OnlineMarketData__{attribute_name}_df_dict")
        attribute_api_get_fn = getattr(self.market_data_end_points, f"get_{attribute_name}")

        symbol_info = self.get_symbol_info(symbol=symbol)
        attribute_get_dataframe_kwargs = {"symbol": symbol, "time_frame": time_frame}
        df = call_with_dict(attribute_get_dataframe, attribute_get_dataframe_kwargs).iloc[:-1]

        # calculate start timestamp
        if 0 == len(df):
            start_timestamp = symbol_info.on_board_timestamp or to_timestamp(date="2019-01-01")
        else:
            start_timestamp = df.index[-1] + time_frame

        # calculate stop timestamp
        current_timestamp = int(datetime.now().timestamp())
        stop_timestamp = ceil_timestamp(timestamp=current_timestamp, time_frame=time_frame)

        attribute_get_fn_kwargs = {
            "symbol": symbol,
            "time_frame": time_frame,
            "start_timestamp": start_timestamp,
            "stop_timestamp": stop_timestamp,
            "progress": progress
        }
        results = call_with_dict(attribute_api_get_fn, attribute_get_fn_kwargs)

        new_df = dtype.to_dataframe(results)
        df = pd.concat([df, new_df]) if 0 != len(new_df) and 0 != len(df) else (df if 0 != len(df) else new_df)

        if check_for_missing_values:
            df = self.__check_time_series_dataframe(
                symbol=symbol,
                time_frame=time_frame,
                df=df,
                attribute_name=attribute_name,
                dtype=dtype,
            )

        self.__save_time_series_dataframe(
            symbol=symbol,
            time_frame=time_frame,
            df=df.iloc[:-1],
            attribute_name=attribute_name,
        )

        attribute_df_dict[(symbol, time_frame)] = df

    def __check_time_series_dataframe(
            self,
            symbol: str,
            time_frame: TimeFrame,
            df: pd.DataFrame,
            attribute_name: str,
            dtype: Type[Union[AggregatedTrade, Candle, FundingRate, LongShortRatio]],
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
            dtype: Type[Union[AggregatedTrade, Candle, FundingRate, LongShortRatio]],
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

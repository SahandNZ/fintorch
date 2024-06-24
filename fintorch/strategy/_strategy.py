import os
import pickle
import filelock
from abc import ABC
from datetime import datetime
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..component import Component
from ..deep.module import Module
from ..dtype import Position, DataCollection
from ..enum import TimeFrame
from ..exchange import Exchange, Market
from ..utils.hash import static_list_hash
from ..utils.directory import create_directory
from ..utils.plot import draw_candlestick_plot
from ..utils.timestamp import floor_timestamp, ceil_timestamp
from ..settings import FINTORCH_STRATEGY_DIR

class Strategy(Component, ABC):
    def __init__(
            self,
            name: str,
            short_name: str,
            module: Module,
            indicators: List[str],
            height_ratios: List[int]
    ):
        super().__init__(name=name, short_name=short_name, description="")
        self.__module: Module = module
        self.__indicators: List[str] = ["trend", "bmax", "bmin"] + indicators
        self.__height_ratios: List[int] = height_ratios

        self.__static_hash: int = static_list_hash([self.short_name, self.module.static_hash])
        self.__directory: str = os.path.join(FINTORCH_STRATEGY_DIR, short_name)
        self.__path: str = os.path.join(self.directory, f"{str(self.static_hash)}.pkl")

        # engine related properties
        self.exchange: Union[Exchange, None] = None

        # process data
        self.__processed_df: Union[pd.DataFrame, None] = None
        self.__df: Union[pd.DataFrame, None] = None

    @property
    def module(self) -> Module:
        return self.__module

    @property
    def indicators(self) -> List[str]:
        return self.__indicators

    @property
    def height_ratios(self) -> List[int]:
        return self.__height_ratios

    @property
    def symbol(self) -> str:
        return self.module.symbol

    @property
    def time_frame(self) -> TimeFrame:
        return self.module.time_frame

    @property
    def time_frames(self) -> List[TimeFrame]:
        return self.module.dataset.time_frames

    @property
    def static_hash(self) -> int:
        return self.__static_hash

    @property
    def directory(self) -> str:
        return self.__directory
    
    @property
    def path(self) -> str:
        return self.__path

    @property
    def future(self) -> Market:
        return self.exchange.future

    @property
    def df(self) -> pd.DataFrame:
        return self.__df

    def preprocess(self, dc: DataCollection = None) -> None:
        if dc is None:
            dc = self.future.data.get_data_collection(symbols=[self.symbol], time_frames=self.time_frames)

        self.__processed_df = self.process_dc_to_df(dc=dc)

    def process_dc_to_df(self, dc: DataCollection) -> pd.DataFrame:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame).copy()
        df.insert(0, "datetime", [datetime.fromtimestamp(ts) for ts in df.index])
        
        y_hat_dict = self.module.predict(timestamps=df.index, dc=dc)
        df["trend"] = [np.argmax(y_hat) if not np.isnan(y_hat).max() else np.nan for y_hat in y_hat_dict.values()]
        df["side"] = np.where(1 == df.trend, 1, np.where(0 == df.trend, -1, np.nan))
        df["toggle"] = df.side.diff()
        
        df["bmax"] = df.high.rolling(window=5).max()
        df["bmin"] = df.low.rolling(window=5).min()
        
        return df

    def on_new_candle(self) -> None:
        pass

    def on_opened_position(self, position: Position) -> None:
        pass

    def on_closed_position(self, position: Position) -> None:
        pass

    def get_state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state_dict: Dict) -> None:
        pass

    def clear_state(self) -> None:
        pass

    def open(self) -> None:
        self.module.open()

        try:
            with open(self.path, "rb") as file:
                state_dict = pickle.load(file)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            state_dict = {}

        self.load_state_dict(state_dict=state_dict)

    def next(self, timestamp: int) -> None:
        current_open_timestamp = floor_timestamp(timestamp=timestamp, time_frame=self.time_frame)
        previous_open_timestamp = current_open_timestamp - int(self.time_frame)
        if previous_open_timestamp not in self.__processed_df.index:
            self.preprocess()

        self.__df = self.__processed_df[self.__processed_df.index < timestamp]

    def close(self) -> None:
        self.module.close()

        create_directory(self.directory)
        with filelock.FileLock(self.path):
            with open(self.path, "wb+") as file:
                pickle.dump(self.get_state_dict(), file)

        self.clear_state()

    def draw_candlestick_and_indicators_plot(
        self,
        dc: DataCollection,
        position: Position,
        interval: TimeFrame,
        look_back: int,
        look_ahead: int
    ) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        entry_timestamp = position.entry_timestamp
        exit_timestamp = position.exit_timestamp or position.current_timestamp

        start_timestamp = entry_timestamp - look_back * int(interval)
        stop_timestamp = exit_timestamp + look_ahead * int(interval)

        start_timestamp = floor_timestamp(timestamp=start_timestamp, time_frame=self.time_frame)
        stop_timestamp = ceil_timestamp(timestamp=stop_timestamp, time_frame=self.time_frame)

        # add indicators and crop dataframe based on timestamps
        df = dc.get_candles_df(symbol=self.symbol, time_frame=interval).copy()
        df = df[(start_timestamp <= df.index) & (df.index < stop_timestamp)]
        pdf = self.process_dc_to_df(dc=dc)
        for indicator in self.indicators:
            df[indicator] = pdf[indicator]
            df[indicator] = df[indicator].ffill()

        # create figure and axis
        fig, axis = plt.subplots(
            nrows=len(self.height_ratios),
            ncols=1,
            sharex=True,
            figsize=(20, 10),
            gridspec_kw={'height_ratios': self.height_ratios}
        )

        # cast output of plt.subplots
        if isinstance(axis, np.ndarray):
            axis = axis.tolist()
        elif isinstance(axis, plt.Axes):
            axis = [axis]
        else:
            raise RuntimeError("Something bad happened when try to cast output of plt.subplots method.")

        # use first row for ohlc plot
        ohlc_ax: plt.Axes = axis[0]

        # draw candlestick subplot
        start_date = datetime.fromtimestamp(df.index[0])
        stop_date = datetime.fromtimestamp(df.index[-1])
        title = "{} - {}\nFrom {} to {}".format(self.symbol, str(interval), start_date, stop_date)
        draw_candlestick_plot(ax=ohlc_ax, df=df)
        ohlc_ax.set_title(title)
        for tick in ohlc_ax.get_xticklabels():
            tick.set_rotation(0)

        # draw indicator subplots
        df.reset_index(inplace=True)
        self._draw_indicators_plot(axis=axis, df=df)
        df.set_index("timestamp", inplace=True)

        return fig, ohlc_ax, df

    def _draw_indicators_plot(self, axis: List[plt.Axes], df: pd.DataFrame) -> None:
        df = df.copy()
        df["y"] = df.low.min()
        udf = df[(1 == df.trend) & (0 == df.timestamp % self.time_frame)]
        ddf = df[(0 == df.trend) & (0 == df.timestamp % self.time_frame)]
        ndf = df[np.isnan(df.trend) & (0 == df.timestamp % self.time_frame)]

        # draw prediction scatters
        ohlc_ax = axis[0]
        size = 2 ** 12 // len(df)
        ohlc_ax.scatter(x=udf.index.to_list(), y=udf.y, s=size, marker='o', c="g", label="Up Prediction")
        ohlc_ax.scatter(x=ddf.index.to_list(), y=ddf.y, s=size, marker='o', c="r", label="Down Prediction")
        ohlc_ax.scatter(x=ndf.index.to_list(), y=ndf.y, s=size, marker='o', c="gray", label="Nan Prediction")
        
        # draw indicators
        ohlc_ax.plot(df.bmin, "y--", label="Backward Minimum (Buy Order Price)")
        ohlc_ax.plot(df.bmax, "m--", label="Backward Maximum (Sell Order Pirce)")
        
        ohlc_ax.legend()

    def __str__(self):
        return f"{str(self.module)} {self.short_name}"

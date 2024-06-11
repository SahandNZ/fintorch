from abc import abstractmethod
from datetime import datetime
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..component import Component
from ..deep.module import Module
from ..dtype import Position, DataCollection
from ..enum import TimeFrame
from ..exchange import Exchange, Market
from ..utils.hash import static_list_hash
from ..utils.plot import draw_candlestick_plot
from ..utils.timestamp import floor_timestamp, ceil_timestamp


class Strategy(Component):
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
        self.__indicators: List[str] = indicators
        self.__height_ratios: List[int] = height_ratios

        self._static_hash: int = static_list_hash([self.short_name, self.module.static_hash])

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
        return self._static_hash

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

    @abstractmethod
    def process_dc_to_df(self, dc: DataCollection) -> pd.DataFrame:
        raise NotImplementedError()

    def on_new_candle(self) -> None:
        pass

    def on_opened_position(self, position: Position) -> None:
        pass

    def on_closed_position(self, position: Position) -> None:
        pass

    def open(self) -> None:
        self.module.open()

    def next(self, timestamp: int) -> None:
        current_open_timestamp = floor_timestamp(timestamp=timestamp, time_frame=self.time_frame)
        previous_open_timestamp = current_open_timestamp - int(self.time_frame)
        if previous_open_timestamp not in self.__processed_df.index:
            self.preprocess()

        self.__df = self.__processed_df[self.__processed_df.index < timestamp]

    def close(self) -> None:
        self.module.close()

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

    @abstractmethod
    def _draw_indicators_plot(self, axis: List[plt.Axes], df: pd.DataFrame) -> None:
        raise NotImplementedError()

    def __str__(self):
        return f"{str(self.module)} {self.short_name}"

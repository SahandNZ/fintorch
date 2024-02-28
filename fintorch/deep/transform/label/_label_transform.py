import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._transform import Transform
from ....dtype import DataCollection
from ....enum import TimeFrame
from ....setting import NUMPY_LABEL_DTYPE
from ....utils.plot import draw_candlestick_plot, draw_labels
from ....utils.timestamp import create_timestamps, to_timestamp


class LabelTransform(Transform, ABC):
    def __init__(
            self,
            name: str,
            short_name: str,
            description: str,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            look_back: int,
            look_ahead: int,
            classes: List[str]
    ):
        super().__init__(
            name=name,
            short_name=short_name,
            description=description,
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=look_back,
            look_ahead=look_ahead
        )
        self.__classes: List[str] = classes

    @property
    def classes(self) -> List[str]:
        return self.__classes

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _shift_timestamp(self, timestamp: int) -> int:
        return math.ceil(timestamp / self.time_frame) * self.time_frame

    def _can_not_be_none(self, dc: DataCollection, timestamp: int) -> bool:
        symbol_info = dc.get_symbol_info(symbol=self.symbol)
        current_open_timestamp = datetime.now().timestamp() // int(self.time_frame) * int(self.time_frame)
        first_timestamp = symbol_info.on_board_timestamp // int(self.time_frame) * self.time_frame
        last_timestamp = current_open_timestamp - int(self.time_frame) * self.look_ahead

        return first_timestamp <= timestamp <= last_timestamp

    @abstractmethod
    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform_df_to_sf(self, df: pd.DataFrame, timestamp: int) -> Union[np.array, None]:
        # forward cropping label dataframe with timestamp and sequence length
        ldf = df[timestamp <= df.index]
        ldf = ldf.iloc[:self.dim_sequence]

        # make sure there is enough time steps and there is no nan values
        if self.dim_sequence != len(ldf) or not np.isnan(ldf.label).max():
            return None

        # one hot encoding
        labels = ldf.label.to_numpy()
        one_hot = np.zeros(self.num_classes)
        one_hot[labels] = 1

        # reshape one hot encoding
        sf = one_hot.astype(dtype=NUMPY_LABEL_DTYPE)
        sf = sf.reshape(self.dim_sequence, 2)

        return sf

    def draw_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str) \
            -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        # create start and stop timestamps
        start_timestamp = to_timestamp(date=start_date)
        stop_timestamp = to_timestamp(date=stop_date)

        # process and crop df based on timestamps
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame).copy()
        df = self._process_df(df=df)
        df = df[(start_timestamp <= df.index.to_series()) & (df.index.to_series() < stop_timestamp)]

        # draw candlestick plot
        fig, ohlc_ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        draw_candlestick_plot(ax=ohlc_ax, df=df)
        draw_labels(ohlc_ax=ohlc_ax, df=df)

        # draw line
        df.reset_index(drop=False, inplace=True)
        self._draw_lines(ohlc_ax=ohlc_ax, df=df)
        df.set_index("timestamp", inplace=True)

        return fig, ohlc_ax, df

    def show_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str) -> None:
        _, ohlc_ax, _ = self.draw_ohlc_plot(dc=dc, start_date=start_date, stop_date=stop_date)

        ohlc_ax.grid()
        ohlc_ax.legend()

        plt.title("{} (from {} to {})".format(str(self), start_date, stop_date))
        plt.show()

    @abstractmethod
    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        raise NotImplementedError()

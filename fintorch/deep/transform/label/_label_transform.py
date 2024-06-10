from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._transform import Transform
from ....dtype import DataCollection
from ....enum import TimeFrame
from ....utils.plot import draw_candlestick_plot, draw_labels
from ....utils.timestamp import to_timestamp, ceil_timestamp


class LabelTransform(Transform, ABC):
    def __init__(
            self,
            name: str,
            short_name: str,
            description: str,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            look_ahead: int,
            look_back: int,
            classes: List[str],
    ):
        super().__init__(
            name=name,
            short_name=short_name,
            description=description,
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            dim_feature=len(classes),
            look_back=look_back,
            look_ahead=look_ahead,
        )
        self.__classes: List[str] = classes
        self.__df: Union[pd.DataFrame, None] = None

    @property
    def classes(self) -> List[str]:
        return self.__classes

    @property
    def df(self) -> Union[pd.DataFrame, None]:
        return self.__df

    def _shift_timestamp(self, timestamp: int) -> int:
        return ceil_timestamp(timestamp=timestamp, time_frame=self.time_frame)

    def _transform_dc_to_sf(self, dc: DataCollection, timestamp: int) -> np.ndarray:
        if self.df is None or timestamp not in self.df.index:
            df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
            df = self.transform_df(df=df)
            df = df.dropna()
            self.__df = df

        return self._transform_df_to_sf(df=self.df, timestamp=timestamp)

    def _transform_df_to_sf(self, df: pd.DataFrame, timestamp: int) -> np.ndarray:
        # forward cropping label dataframe with timestamp and sequence length
        ldf = df[timestamp <= self.df.index]
        ldf = ldf.iloc[: self.dim_sequence]

        # make sure there is enough time steps and there is no nan values
        if self.dim_sequence != len(ldf):
            return np.zeros((self.dim_sequence, self.dim_feature)) * np.nan

        # one hot encoding
        label = ldf.label.to_numpy().astype(int)
        sf = np.zeros((self.dim_sequence, self.dim_feature))
        sf[range(self.dim_sequence), label] = 1

        return sf

    def draw_ohlc_plot(
            self,
            dc: DataCollection,
            start_date: Union[str, datetime],
            stop_date: Union[str, datetime]
    ) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
        # create start and stop timestamps
        start_timestamp = to_timestamp(date=start_date)
        stop_timestamp = to_timestamp(date=stop_date)

        # process and crop df based on timestamps
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame).copy()
        df = self.transform_df(df=df)
        df = df[(start_timestamp <= df.index.to_series()) & (df.index.to_series() < stop_timestamp)]

        # draw candlestick plot
        fig, ohlc_ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        draw_candlestick_plot(ax=ohlc_ax, df=df)
        draw_labels(ohlc_ax=ohlc_ax, df=df)

        # draw line
        df.reset_index(drop=False, inplace=True)
        self._draw_lines(ohlc_ax=ohlc_ax, df=df)
        df.set_index("timestamp", inplace=True)

        # set title and legend
        fig.suptitle(str(self))
        ohlc_ax.legend()

        return fig, ohlc_ax, df

    def show_ohlc_plot(self, dc: DataCollection, start_date: str, stop_date: str) -> None:
        _, _, _ = self.draw_ohlc_plot(dc=dc, start_date=start_date, stop_date=stop_date)
        plt.show()

    @abstractmethod
    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        raise NotImplementedError()

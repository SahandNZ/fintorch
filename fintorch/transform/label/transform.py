from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fintorch.data import Data
from fintorch.plot import draw_ohlcv_plot
from fintorch.transform.transform import Transform


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbol: str, time_frame: int, look_ahead: int,
                 num_classes: int = None):
        super().__init__(name, short_name, description)
        self.__symbol: str = symbol
        self.__time_frame: int = time_frame
        self.__look_ahead: int = look_ahead
        self.__num_classes: int = num_classes

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def time_frame(self) -> int:
        return self.__time_frame

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    def fit(self, data: Data) -> pd.DataFrame:
        df = data[self.symbol, self.time_frame].copy()
        return self._fit(df=df)

    def transform(self, df: pd.DataFrame, timestamps: List[int]) -> List[np.array]:
        labels = []
        for timestamp in timestamps:
            label = self._transform(df=df, timestamp=timestamp)
            labels.append(label)
        return labels

    def draw_ohlcv_plot(self, df: pd.DataFrame, prediction: np.array = None, volume: bool = False) -> plt.Figure:
        df = df.copy()
        ldf = self._fit(df=df.copy()).drop(columns=df.columns)
        df = df.join(other=ldf, how="left")
        df["prediction"] = prediction if prediction is not None else np.nan

        df = df.reset_index()
        fig, ohlcv_ax, volume_ax = draw_ohlcv_plot(df=df, volume=volume)
        self._draw_lines(df=df, ohlcv_ax=ohlcv_ax, volume_ax=volume_ax)

        return fig

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform(self, df: pd.DataFrame, timestamp: int) -> np.array:
        if timestamp in df.index:
            label = df.loc[timestamp].label
            # regression
            if self.num_classes is None:
                return label.to_numpy()

            # classification
            else:
                one_hot = np.zeros(self.num_classes)
                one_hot[label] = 1
                return one_hot
        else:
            if self.num_classes is None:
                return np.nan
            else:
                return [np.nan] * self.num_classes

    @abstractmethod
    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        raise NotImplementedError()

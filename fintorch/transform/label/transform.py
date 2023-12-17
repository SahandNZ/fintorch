import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fintorch.data import Data
from fintorch.transform.transform import Transform
from fintorch.utils.plot import draw_ohlcv_plot


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbol: str, time_frame: int, look_ahead: int,
                 num_classes: int = None):
        super().__init__(name, short_name, description)
        self.__symbol: str = symbol
        self.__time_frame: int = time_frame
        self.__look_ahead: int = look_ahead
        self.__num_classes: int = num_classes

        # temporal states
        self.__dataframe: pd.DataFrame = None

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

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__dataframe

    def fit(self, data: Data) -> pd.DataFrame:
        raw_dataframe = data[self.symbol, self.time_frame].copy()
        self.__dataframe = self._fit(df=raw_dataframe)

    def transform(self, timestamps: List[int]) -> Dict[int, np.array]:
        labels_dict = {}
        for timestamp in timestamps:
            forward_timestamp = math.ceil(timestamp / self.time_frame) * self.time_frame
            label = self._transform(timestamp=forward_timestamp)
            if label is not None:
                labels_dict[timestamp] = label

        return labels_dict

    def fit_transform(self, data: Data, timestamps: List[int]) -> Dict[int, np.array]:
        self.fit(data=data)
        return self.transform(timestamps=timestamps)

    def draw_ohlcv_plot(self, df: pd.DataFrame, prediction: np.array = None, volume: bool = False,
                        figsize: Tuple[float, float] = (20, 10)) -> plt.Figure:
        df = df.copy()
        ldf = self._fit(df=df.copy()).drop(columns=df.columns)
        df = df.join(other=ldf, how="left")
        df["prediction"] = prediction if prediction is not None else np.nan

        df = df.reset_index()
        fig, ohlcv_ax, volume_ax = draw_ohlcv_plot(df=df, volume=volume, figsize=figsize)
        self._draw_lines(df=df, ohlcv_ax=ohlcv_ax, volume_ax=volume_ax)

        return fig

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform(self, timestamp: int) -> Union[np.array, None]:
        if timestamp in self.__dataframe.index:
            label = self.__dataframe.loc[timestamp].label
            # regression
            if self.num_classes is None:
                return label.to_numpy()

            # classification
            else:
                one_hot = np.zeros(self.num_classes)
                one_hot[label] = 1
                return one_hot
        else:
            return None

    @abstractmethod
    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        raise NotImplementedError()

    def __eq__(self, other):
        is_symbols_equal = self.symbol == other.symbol
        is_time_frames_equal = self.time_frame == other.time_frame
        is_look_ahead_equal = self.look_ahead == other.look_ahead

        return super().__eq__(other) and is_symbols_equal and is_time_frames_equal and is_look_ahead_equal

    def __getstate__(self):
        dct = self.__dict__.copy()
        if "_LabelTransform__dataframe" in dct:
            del dct["_LabelTransform__dataframe"]

        return dct

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from fintorch.dataset.data import Data
from fintorch.transform.transform import Transform


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbol: str, time_frame: int,
                 num_classes: int = None):
        super().__init__(name, short_name, description)
        self.__symbol: str = symbol
        self.__time_frame: int = time_frame
        self.__num_classes: int = num_classes

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def time_frame(self) -> int:
        return self.__time_frame

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

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform(self, df: pd.DataFrame, timestamp: List[int]) -> np.array:
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

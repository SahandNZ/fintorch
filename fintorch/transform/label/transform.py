from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

from fintorch.dataset.data import Data
from fintorch.transform.transform import Transform


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbol: str, time_frame: int,
                 label2side: Dict[int, int] = None):
        super().__init__(name, short_name, description)
        self.__symbol: str = symbol
        self.__time_frame: int = time_frame
        self.__label2side: Dict[int, int] = label2side

    @property
    def symbol(self) -> str:
        return self.__symbol

    @property
    def time_frame(self) -> int:
        return self.__time_frame

    @property
    def label2side(self) -> Dict[int, int]:
        return self.__label2side

    @property
    def num_classes(self) -> int:
        return len(self.label2side) if self.label2side else None

    def fit(self, data: Data) -> pd.DataFrame:
        df = data[self.symbol, self.time_frame].copy()
        return self._fit(df=df)

    def transform(self, df: pd.DataFrame, timestamp: int) -> np.array:
        return self._transform(df=df, timestamp=timestamp)

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform(self, df: pd.DataFrame, timestamp: int) -> np.array:
        label = df[df.index.to_series() == timestamp].label
        if 0 < len(label):
            # regression
            if self.num_classes is None:
                return label

            # classification
            else:
                one_hot = np.zeros(self.num_classes)
                one_hot[label] = 1
                return one_hot

        else:
            return None

    def to_side(self, label: int):
        return self.label2side[label]

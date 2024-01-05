import itertools
import math
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd

from fintorch.data import Data
from fintorch.transform.transform import Transform


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, look_ahead: int, classes: List[str]):
        super().__init__(name, short_name, description)
        self.__look_ahead: int = look_ahead
        self.__classes: int = classes

        self.__data: Data = None

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    @property
    def classes(self) -> List[str]:
        return self.__classes

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def data(self) -> Data:
        return self.__data

    def fit(self, data: Data) -> None:
        self.__data = Data()
        for symbol, time_frame in itertools.product(data.symbols, data.time_frames):
            df = data[symbol, time_frame].copy()
            df = self._fit(df)
            self.__data[symbol, time_frame] = df

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def transform(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        ldf = self.__data[symbol, time_frame]
        forward_timestamp = math.ceil(timestamp / time_frame) * time_frame
        if forward_timestamp not in ldf.index:
            return None

        # one hot encoding
        label = ldf.label.loc[forward_timestamp]
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1

        return one_hot

    def __getstate__(self):
        dct = self.__dict__.copy()
        if "_LabelTransform__data" in dct:
            del dct["_LabelTransform__data"]

        return dct

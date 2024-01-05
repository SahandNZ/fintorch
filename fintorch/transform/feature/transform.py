import itertools
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd

from fintorch.data import Data
from fintorch.transform.transform import Transform


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, look_back: int, features: List[str]):
        super().__init__(name, short_name, description="")
        self.__look_back: int = look_back
        self.__features: List[str] = features

        self.__data: Data = None

    @property
    def look_back(self) -> int:
        return self.__look_back

    @property
    def features(self) -> List[str]:
        return self.__features

    @property
    def data(self) -> Data:
        return self.__data

    def fit(self, data: Data) -> None:
        self.__data = Data()
        for symbol, time_frame in itertools.product(data.symbols, data.time_frames):
            df = data[symbol, time_frame].copy()
            df = self._fit(df)
            df = df[self.features]
            self.__data[symbol, time_frame] = df

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    def transform(self, timestamp: int, symbol: str, time_frame: int, sequence_length: int) -> Union[np.array, None]:
        fdf = self.__data[symbol, time_frame]
        fdf = fdf[fdf.index < timestamp]
        fdf = fdf.iloc[-sequence_length:]

        if sequence_length != len(fdf):
            return None

        ndf = fdf / fdf.std()
        sf = ndf.to_numpy()

        return sf

    def __getstate__(self):
        dct = self.__dict__.copy()
        if "_FeatureTransform__data" in dct:
            del dct["_FeatureTransform__data"]

        return dct

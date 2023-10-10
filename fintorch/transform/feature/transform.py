import copy
import itertools
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from fintorch.dataset.data import Data
from fintorch.transform.transform import Transform


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbols: List[str], time_frames: int,
                 features: List[str], sequence_length: int):
        super().__init__(name, short_name, description)
        self.__symbols: List[str] = symbols
        self.__time_frames: List[int] = time_frames
        self.__features: List[str] = features
        self.__sequence_length: int = sequence_length

    @property
    def symbols(self) -> List[str]:
        return self.__symbols

    @property
    def time_frames(self) -> List[int]:
        return self.__time_frames

    @property
    def features(self) -> List[str]:
        return self.__features

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    def fit(self, data: Data):
        feature_data = copy.deepcopy(data)
        for symbol, time_frame in itertools.product(self.symbols, self.time_frames):
            df = feature_data[symbol, time_frame]
            df = self._fit(df)
            feature_data[symbol, time_frame] = df

        return feature_data

    def transform(self, data: Data, timestamp: int) -> np.array:
        atf = []
        for symbol in self.symbols:
            tf = []
            for time_frame in self.time_frames:
                df = data[symbol, time_frame]
                f = self._transform(df=df, timestamp=timestamp)

                if f is None:
                    return None
                else:
                    tf.append(f)

            atf.append(tf)

        return np.array(atf)

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    def _transform(self, df: pd.DataFrame, timestamp: int) -> np.array:
        df = df[df.index.to_series() < timestamp]
        df = df[self.features]
        if self.sequence_length <= len(df):
            df = df.iloc[-self.sequence_length:]
            df = df / df.std()

            feature = df.to_numpy()

            return feature
        else:
            return None

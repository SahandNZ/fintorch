import math
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd

from fintorch.defaults import NUMPY_FEATURE_DTYPE
from fintorch.transform.transform import Transform


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, look_back: int, sequence_length: int, features: List[str]):
        super().__init__(name=name, short_name=short_name, description="", sequence_length=sequence_length)
        self.__look_back: int = look_back
        self.__sequence_length: int = sequence_length
        self.__features: List[str] = features

    @property
    def look_back(self) -> int:
        return self.__look_back

    @property
    def features(self) -> List[str]:
        return self.__features

    @property
    def _save_none(self) -> bool:
        return True

    @abstractmethod
    def _fit(self, df: pd.DataFrame):
        raise NotImplementedError()

    def _shift_timestamp(self, timestamp: int, time_frame: int) -> int:
        return math.floor(timestamp / time_frame) * time_frame

    def _transform(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        fdf = self.data[symbol, time_frame]
        fdf = fdf[fdf.index < timestamp]
        fdf = fdf.iloc[-self.sequence_length:]
        fdf = fdf[self.features]

        if self.sequence_length != len(fdf):
            return None

        ndf = fdf / fdf.std()
        sf = ndf.to_numpy().astype(dtype=NUMPY_FEATURE_DTYPE)

        return sf

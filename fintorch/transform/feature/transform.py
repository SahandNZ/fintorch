from abc import ABC
from typing import List

import numpy as np
import pandas as pd

from fintorch.transform.transform import Transform


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, features: List[str], sequence_length: int):
        super().__init__(name, short_name, description)
        self.__features: List[str] = features
        self.__sequence_length: int = sequence_length

    @property
    def features(self) -> List[str]:
        return self.__features

    @property
    def sequence_length(self) -> int:
        return self.__sequence_length

    def transform(self, df: pd.DataFrame, timestamp: int) -> np.array:
        df = df[df.index.to_series() < timestamp]
        df = df[self.features]
        df = df.iloc[-self.sequence_length:]
        df = df / df.std()

        feature = df.to_numpy()

        return feature

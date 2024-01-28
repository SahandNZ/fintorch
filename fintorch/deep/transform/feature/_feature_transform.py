import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

from .._transform import Transform
from ....dtype import Data
from ....setting import NUMPY_FEATURE_DTYPE


class FeatureTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, sequence_length: int, look_back: int, features: List[str]):
        super().__init__(name=name, short_name=short_name, description="", sequence_length=sequence_length)
        self.__sequence_length: int = sequence_length
        self.__look_back: int = look_back
        self.__features: List[str] = features

    @property
    def look_back(self) -> int:
        return self.__look_back

    @property
    def features(self) -> List[str]:
        return self.__features

    def _shift_timestamp(self, timestamp: int, time_frame: int) -> int:
        return math.floor(timestamp / time_frame) * time_frame

    def _is_none_possible(self, timestamp: int) -> bool:
        return timestamp < datetime.strptime("2022-01-01", "%Y-%m-%d").timestamp()

    @abstractmethod
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform_dataframe(self, df: pd.DataFrame, timestamp: int) -> Union[np.array, None]:
        # backward cropping feature dataframe with timestamp and sequence length
        fdf = df[df.index < timestamp]
        fdf = fdf.iloc[-self.sequence_length:]
        fdf = fdf[self.features]

        if self.sequence_length != len(fdf):
            return None

        # z-score standardization and min-max normalization (keep negative values)
        zdf = (fdf - fdf.mean()) / fdf.std()
        ndf = zdf / (zdf.max() - zdf.min())
        sf = ndf.to_numpy().astype(dtype=NUMPY_FEATURE_DTYPE)

        return sf

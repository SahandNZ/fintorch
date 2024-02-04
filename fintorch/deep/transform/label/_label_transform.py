import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, List

import numpy as np
import pandas as pd

from .._transform import Transform
from ....enum import TimeFrame
from ....setting import NUMPY_LABEL_DTYPE


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, symbol: str, time_frame: TimeFrame,
                 dim_sequence: int, look_ahead: int, classes: List[str]):
        super().__init__(name=name, short_name=short_name, description=description, symbol=symbol,
                         time_frame=time_frame, dim_sequence=dim_sequence)
        self.__look_ahead: int = look_ahead
        self.__classes: List[str] = classes

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    @property
    def classes(self) -> List[str]:
        return self.__classes

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _shift_timestamp(self, timestamp: int) -> int:
        return math.ceil(timestamp / self.time_frame) * self.time_frame

    def _can_be_none(self, timestamp: int) -> bool:
        return datetime.strptime("2022-01-01", "%Y-%m-%d").timestamp() < timestamp

    @abstractmethod
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _transform_dataframe(self, df: pd.DataFrame, timestamp: int) -> Union[np.array, None]:
        # forward cropping label dataframe with timestamp and sequence length
        ldf = df[timestamp <= df.index]
        ldf = ldf.iloc[:self.dim_sequence]

        if self.dim_sequence != len(ldf):
            return None

        # one hot encoding
        labels = ldf.label.to_numpy()
        one_hot = np.zeros(self.num_classes)
        one_hot[labels] = 1

        # reshape one hot encoding
        sf = one_hot.astype(dtype=NUMPY_LABEL_DTYPE)
        sf = sf.reshape(self.dim_sequence, 2)

        return sf

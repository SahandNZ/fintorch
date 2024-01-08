import math
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd

from fintorch.defaults import NUMPY_LABEL_DTYPE
from fintorch.transform.transform import Transform


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, short_name: str, description: str, look_ahead: int, classes: List[str]):
        super().__init__(name=name, short_name=short_name, description=description, sequence_length=1)
        self.__look_ahead: int = look_ahead
        self.__classes: int = classes

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
    def _save_none(self) -> bool:
        return False

    @abstractmethod
    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _shift_timestamp(self, timestamp: int, time_frame: int) -> int:
        return math.ceil(timestamp / time_frame) * time_frame

    def _transform(self, timestamp: int, symbol: str, time_frame: int) -> Union[np.array, None]:
        ldf = self.data[symbol, time_frame]
        if timestamp not in ldf.index:
            return None

        # one hot encoding
        label = ldf.label.loc[timestamp]
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1

        one_hot = one_hot.astype(dtype=NUMPY_LABEL_DTYPE)

        return one_hot

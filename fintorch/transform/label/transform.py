from abc import ABC

import numpy as np
import pandas as pd

from fintorch.transform.transform import Transform


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, num_classes: int):
        super().__init__(name)
        self.__num_classes: int = num_classes

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    def transform(self, df: pd.DataFrame, timestamp: int) -> np.array:
        label_series = df.label - df.label.min()
        label = label_series[label_series.index.to_series() == timestamp]
        if 1 < self.num_classes:
            one_hot = np.zeros(self.num_classes)
            one_hot[label] = 1
            label = one_hot

        return label

from abc import ABC

import numpy as np
import pandas as pd
from fintorch.transform.transform import Transform


class LabelTransform(Transform, ABC):
    def __init__(self, name: str, num_classes: int, min_class: int):
        super().__init__(name)
        self.__num_classes: int = num_classes
        self.__min_class: int = min_class

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def min_class(self) -> int:
        return self.__min_class

    def transform(self, df: pd.DataFrame, timestamp: int) -> np.array:
        label = df[df.index.to_series() == timestamp].label
        if 0 < len(label):
            # one hot encoding
            if self.num_classes is not None:
                label = label - self.min_class
                one_hot = np.zeros(self.num_classes)
                one_hot[label] = 1
                label = one_hot

            return label
        else:
            return None

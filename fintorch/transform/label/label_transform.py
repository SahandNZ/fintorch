from abc import abstractmethod

import pandas as pd

from fintorch.transform.transform import Transform


class LabelTransform(Transform):
    def __init__(self, name: str):
        super().__init__(name)

    def fit(self, df: pd.DataFrame):
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        raise NotImplementedError()

    def inverse_transform(self, df: pd.DataFrame):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

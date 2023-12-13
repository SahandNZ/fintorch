from abc import abstractmethod

import numpy as np
import pandas as pd

from fintorch.component import Component


class Transform(Component):
    def __init__(self, name: str, short_name: str, description: str):
        super().__init__(name, short_name, description)

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        NotImplemented()

    @abstractmethod
    def transform(self, *args) -> np.array:
        NotImplemented()

    @abstractmethod
    def fit_transform(self, *args):
        NotImplemented()

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

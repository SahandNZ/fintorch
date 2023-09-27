from abc import abstractmethod

import numpy as np
import pandas as pd


class Transform:
    def __init__(self, name: str):
        self.__name: str = name

    @property
    def name(self) -> str:
        return self.__name

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        NotImplemented()

    @abstractmethod
    def transform(self, *args) -> np.array:
        NotImplemented()

    def __str__(self):
        return self.name

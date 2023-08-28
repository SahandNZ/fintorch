from abc import abstractmethod

import pandas as pd


class Transform:
    def __init__(self, name: str):
        self.name: str = name

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        NotImplemented()

    @abstractmethod
    def transform(self, df: pd.DataFrame):
        NotImplemented()

    def fit_transform(self, dataframe: pd.DataFrame):
        self.fit(dataframe)
        return self.transform(dataframe)

    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame):
        NotImplemented()

    @abstractmethod
    def save(self, path: str):
        NotImplemented()

    @abstractmethod
    def load(self, path: str):
        NotImplemented()

    def __str__(self):
        return self.name

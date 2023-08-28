import pandas as pd

from fintorch.transform.transform import Transform


class MinMaxNormalizationTransform(Transform):
    def __init__(self):
        super().__init__(name='Min Max Normalization Transform')
        self.__min: pd.Series = None
        self.__max: pd.Series = None

    @property
    def min(self) -> pd.Series:
        return self.__min

    @property
    def max(self) -> pd.Series:
        return self.__max

    def fit(self, df: pd.DataFrame):
        self.__min = df.min()
        self.__max = df.max()

    def transform(self, df: pd.DataFrame):
        return (df - self.min) / (self.max - self.min)

    def inverse_transform(self, df: pd.DataFrame):
        return df * (self.max - self.min) + self.min

    def load(self, path: str):
        df = pd.read_csv(path, index_col=0)
        self.__min = df.MIN
        self.__max = df.MAX

    def save(self, path: str):
        df = pd.DataFrame({'MIN': self.min, 'MAX': self.max})
        df.to_csv(path)

import pandas as pd

from fintorch.transform.transform import Transform


class ZScoreNormalizationTransform(Transform):
    def __init__(self):
        super().__init__(name='Z Score Normalization Transform')
        self.__mean: pd.Series = None
        self.__std: pd.Series = None

    @property
    def mean(self) -> pd.Series:
        return self.__mean

    @property
    def std(self) -> pd.Series:
        return self.__std

    def fit(self, df: pd.DataFrame):
        self.__mean = df.mean()
        self.__std = df.std()

    def transform(self, df: pd.DataFrame):
        return (df - self.mean) / self.std

    def inverse_transform(self, df: pd.DataFrame):
        return df * self.std + self.mean

    def load(self, path: str):
        df = pd.read_csv(path, index_col=0)
        self.__mean = df.MEAN
        self.__std = df.STD

    def save(self, path: str):
        df = pd.DataFrame({'MEAN': self.mean, 'STD': self.std})
        df.to_csv(path)

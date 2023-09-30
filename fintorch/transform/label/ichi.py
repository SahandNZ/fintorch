import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class IchiLabelTransform(LabelTransform):
    def __init__(self, base_length: int = 5, conversion_length: int = 20):
        super().__init__(name="F-Ichi", num_classes=3, min_class=-1)
        self.__base_length: int = base_length
        self.__conversion_length: int = conversion_length

    @property
    def base_length(self) -> int:
        return self.__base_length

    @property
    def conversion_length(self) -> int:
        return self.__conversion_length

    @staticmethod
    def donchian(df: pd.DataFrame, length: int) -> pd.Series:
        return (df.close.rolling(length).max() + df.close.rolling(length).min()) / 2

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df['base'] = self.donchian(df, length=self.base_length)
        df['conversion'] = self.donchian(df, length=self.conversion_length)

        df['up'] = df.conversion < df.base
        df['label'] = np.where(df.up, 1, -1)
        df = df.dropna()

        return df

import numpy as np
import pandas as pd

from fintorch.transform.label.label_transform import LabelTransform


class IchimokuLabelTransform(LabelTransform):
    def __init__(self, base_length: int = 5, conversion_length: int = 20):
        super().__init__(name="Ichimoku")
        self.__base_length: int = base_length
        self.__conversion_length: int = conversion_length

    @property
    def base_length(self) -> int:
        return self.__base_length

    @property
    def conversion_length(self) -> int:
        return self.__conversion_length

    @property
    def num_classes(self) -> int:
        return 3

    @staticmethod
    def donchian(df: pd.DataFrame, length: int) -> pd.Series:
        return (df.close.rolling(length).max() + df.close.rolling(length).min()) / 2

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['base'] = self.donchian(df, length=self.base_length)
        df['conversion'] = self.donchian(df, length=self.conversion_length)
        df['up'] = df.conversion < df.base
        df['down'] = df.base < df.conversion

        df['roc'] = df.close / df.open - 1
        df['label'] = np.where(df.up, 1, np.where(df.down, -1, 0))

        return df

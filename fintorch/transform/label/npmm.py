import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class NpmmLabelTransform(LabelTransform):
    def __init__(self, length: int = 10):
        super().__init__(name='NPMM', num_classes=3, min_class=-1)
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def fit(self, df: pd.DataFrame):
        df['bmin'] = df.close.rolling(self.length).min()
        df['bmax'] = df.close.rolling(self.length).max()
        df['fmin'] = df.close.rolling(self.length).min().shift(-self.length + 1)
        df['fmax'] = df.close.rolling(self.length).max().shift(-self.length + 1)

        df['ispl'] = (df.close == df.fmin) & (df.close == df.bmin)
        df['isph'] = (df.close == df.fmax) & (df.close == df.bmax)
        df['label'] = np.where(df.ispl, 1, np.where(df.isph, -1, 1))
        df = df.dropna()

        return df

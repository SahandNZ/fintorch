import numpy as np
import pandas as pd

from fintorch.transform.label.label_transform import LabelTransform


class NpmmLabelTransform(LabelTransform):
    def __init__(self, period: int = 20):
        super().__init__(name='NPMM')
        self.__period: int = period

    @property
    def period(self) -> int:
        return self.__period

    @property
    def num_classes(self) -> int:
        return 3

    def transform(self, df: pd.DataFrame):
        df['fmax'] = df.high.rolling(self.period).max().shift(-self.period + 1)
        df['fmin'] = df.low.rolling(self.period).min().shift(-self.period + 1)
        df['bmax'] = df.high.rolling(self.period).max()
        df['bmin'] = df.low.rolling(self.period).max()

        df['isph'] = (df.high == df.fmax) & (df.high == df.bmax)
        df['ispl'] = (df.low == df.fmin) & (df.low == df.bmin)

        df['roc'] = df.close / df.open - 1
        df['label'] = np.where(df.isph, 0, np.where(df.ispl, 1, np.nan))

        return df

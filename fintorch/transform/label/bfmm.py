import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class BfmmLabelTransform(LabelTransform):
    def __init__(self, look_back: int = 10, look_ahead: int = 10):
        super().__init__(name='BFMM', num_classes=3)
        self.__look_back: int = look_back
        self.__look_ahead: int = look_ahead

    @property
    def look_back(self) -> int:
        return self.__look_back

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    def fit(self, df: pd.DataFrame):
        df['bmin'] = df.clsoe.rolling(self.look_back).min()
        df['bmax'] = df.close.rolling(self.look_back).max()
        df['fmin'] = df.close.rolling(self.look_ahead).min().shift(-self.look_ahead + 1)
        df['fmax'] = df.close.rolling(self.look_ahead).max().shift(-self.look_ahead + 1)

        df['label'] = np.where(df.bmin <= df.fmin, 1, np.where(df.fmax <= df.bmax, -1, 0))
        df = df.dropna()

        return df

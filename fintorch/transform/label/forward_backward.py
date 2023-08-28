import numpy as np
import pandas as pd

from fintorch.transform.label.label_transform import LabelTransform


class ForwardBackwardLabelTransform(LabelTransform):
    def __init__(self, look_backward: int = 10, look_forward: int = 10):
        super().__init__(name='Forward Backward')
        self.__look_backward: int = look_backward
        self.__look_forward: int = look_forward

    @property
    def look_backward(self) -> int:
        return self.__look_backward

    @property
    def look_forward(self) -> int:
        return self.__look_forward

    @property
    def num_classes(self) -> int:
        return 3

    def transform(self, df: pd.DataFrame):
        df['bmax'] = df.high.rolling(self.look_backward).max()
        df['bmin'] = df.low.rolling(self.look_backward).max()
        df['fmax'] = df.high.rolling(self.look_forward).max().shift(-self.look_forward + 1)
        df['fmin'] = df.low.rolling(self.look_forward).min().shift(-self.look_forward + 1)

        df['roc'] = df.close / df.open - 1
        df['label'] = np.where(df.bmin <= df.fmin, 2, np.where(df.fmax <= df.bmax, 0, 1))

        return df

import pandas as pd

from fintorch.transform.label.label_transform import LabelTransform


class EnpmmLabelTransform(LabelTransform):
    def __init__(self, period: int = 20, extend_left: int = 5, extend_right: int = 5):
        super().__init__(name='ENPMM')
        self.__period: int = period
        self.__extend_left: int = extend_left
        self.__extend_right: int = extend_right

    @property
    def period(self) -> int:
        return self.__period

    @property
    def extend_right(self) -> int:
        return self.__extend_right

    @property
    def extend_left(self) -> int:
        return self.__extend_left

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

        label = [0] * len(df)
        for index in range(len(df)):
            pivot = 1 if df.isph.iloc[index] else -1 if df.ispl.iloc[index] else False
            if pivot:
                for j in range(-self.extend_left, self.extend_right + 1):
                    label[index + j] = pivot

        df['roc'] = df.close / df.open - 1
        df['label'] = label

        return df

import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class PivotLabelTransform(LabelTransform):
    def __init__(self, length: int = 10):
        super().__init__(name='F-Pivot', num_classes=2)
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
        df['isp'] = df.ispl | df.isph

        df['pivot'] = df.close[df.isp]
        df['f-pivot'] = df.pivot.shift(-1).bfill()
        df['label'] = df.close < df['f-pivot']
        df['label'] = df.label.astype(int)
        df = df.dropna()

        return df

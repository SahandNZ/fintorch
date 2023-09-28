import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class FrocLabelTransform(LabelTransform):
    def __init__(self, look_ahead: int = 12):
        super().__init__(name='Forward ROC', num_classes=2)
        self.__look_ahead: int = look_ahead

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    def fit(self, df: pd.DataFrame):
        df['roc'] = df.close / df.open - 1
        df['f-roc'] = df.roc.rolling(self.look_ahead).sum().shift(-self.look_ahead)
        df['label'] = 0 < df['f-roc']
        df['label'] = df.label.astype(int)
        df = df.dropna()

        return df

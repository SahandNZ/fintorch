import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class UpDownLabelTransform(LabelTransform):
    def __init__(self):
        super().__init__(name='Up-Down', num_classes=2)

    @property
    def num_classes(self) -> int:
        return 2

    def fit(self, df: pd.DataFrame):
        df['label'] = df.open < df.close
        df['label'] = df.label.astype(int)
        df = df.dropna()

        return df

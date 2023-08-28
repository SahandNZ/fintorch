import pandas as pd

from fintorch.transform.label.label_transform import LabelTransform


class UpDownLabelTransform(LabelTransform):
    def __init__(self):
        super().__init__(name='UpDown')

    @property
    def num_classes(self) -> int:
        return 2

    def transform(self, df: pd.DataFrame):
        df['roc'] = df.close / df.open - 1
        df['label'] = df.open < df.close

        return df

import pandas as pd

from fintorch.transform.label.label_transform import LabelTransform


class RocLabelTransform(LabelTransform):
    def __init__(self):
        super().__init__(name='ROC')

    @property
    def num_classes(self) -> int:
        return 1

    def transform(self, df: pd.DataFrame):
        df['roc'] = df.close / df.open - 1
        df['label'] = df.roc / df.roc.std()

        return df

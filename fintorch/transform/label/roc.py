import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class RocLabelTransform(LabelTransform):
    def __init__(self):
        super().__init__(name="Rate of Change", short_name="ROC", description="", num_classes=None, min_class=None)

    def fit(self, df: pd.DataFrame):
        df["roc"] = df.close / df.open - 1
        df["label"] = df.roc / df.roc.std()
        df = df.dropna()

        return df

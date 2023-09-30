import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class UpDownLabelTransform(LabelTransform):
    def __init__(self):
        super().__init__(name='Up-Down', num_classes=3, min_class=-1)

    def fit(self, df: pd.DataFrame):
        df['up'] = df.open < df.close
        df['label'] = np.where(df.up, 1, -1)
        df = df.dropna()

        return df

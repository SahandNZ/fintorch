import pandas as pd

from fintorch.transform.transform import Transform


class RocFeatureTransform(Transform):
    def __init__(self):
        super().__init__(name='ROC')

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame):
        df['roc'] = df.close / df.open - 1

        return df.roc

    def inverse_transform(self, df: pd.DataFrame):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

import pandas as pd

from fintorch.transform.feature.transform import FeatureTransform


class RocFeatureTransform(FeatureTransform):
    def __init__(self, sequence_length: int):
        super().__init__(name='ROC', features=['roc'], sequence_length=sequence_length)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df['roc'] = df.close / df.open - 1
        df = df.dropna()
        
        return df

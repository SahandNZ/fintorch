from typing import List

import pandas as pd

from fintorch.transform.feature.transform import FeatureTransform


class RocFeatureTransform(FeatureTransform):
    def __init__(self, symbols: List[str], time_frames: List[int], sequence_length: int = 32):
        super().__init__(
            name="Rate Of Change",
            short_name="ROC",
            description="",
            symbols=symbols,
            time_frames=time_frames,
            features=["roc"],
            sequence_length=sequence_length,
        )

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["roc"] = df.close / df.open - 1
        df = df.dropna()

        return df

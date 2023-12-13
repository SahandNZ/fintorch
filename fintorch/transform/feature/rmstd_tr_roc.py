from typing import List

import pandas as pd

from fintorch.transform.feature.transform import FeatureTransform


class RollingMeanStdTrRocFeatureTransform(FeatureTransform):
    def __init__(self, symbols: List[str], time_frames: List[int], look_back: int = 4, sequence_length: int = 32):
        super().__init__(
            name="Rolling Mean and Standard deviation of True Range and Rate Of Change",
            short_name="RMSTD",
            description="",
            symbols=symbols,
            time_frames=time_frames,
            sequence_length=sequence_length,
            features=["mean-tr", "mean-roc", "std-tr", "std-roc"],
        )
        self.__look_back: int = look_back

    @property
    def look_back(self) -> int:
        return self.__look_back

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["mean-tr"] = df.tr.rolling(self.look_back).mean()
        df["mean-roc"] = df.roc.rolling(self.look_back).mean()
        df["std-tr"] = df.tr.rolling(self.look_back).std()
        df["std-roc"] = df.roc.rolling(self.look_back).std()
        df = df.dropna()

        return df

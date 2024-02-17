import pandas as pd

from ....enum import TimeFrame
from ._feature_transform import FeatureTransform


class RollingMeanStdTrRocFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, look_back: int = 4):
        super().__init__(
            name="Rolling Mean and Standard deviation of True Range and Rate Of Change",
            short_name="RMSTD",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=look_back,
            features=["mean-tr", "mean-roc", "std-tr", "std-roc"],
        )

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["mean-tr"] = df.tr.rolling(self.look_back).mean()
        df["mean-roc"] = df.roc.rolling(self.look_back).mean()
        df["std-tr"] = df.tr.rolling(self.look_back).std()
        df["std-roc"] = df.roc.rolling(self.look_back).std()
        df = df.dropna()

        return df

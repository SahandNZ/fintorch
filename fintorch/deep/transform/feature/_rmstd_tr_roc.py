import pandas as pd

from ._feature_transform import FeatureTransform
from ....enum import TimeFrame
from ....utils.preprocess import remove_price_dependency


class RollingMeanStdTrRocFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, backward: int = 5):
        super().__init__(
            name="Rolling Mean and Standard deviation of True Range and Rate Of Change",
            short_name="RMSTD",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=(dim_sequence + backward) * 2,
            features=["mean-tr", "std-tr", "mean-roc", "std-roc"],
        )
        self.__backward: int = backward

    @property
    def backward(self) -> int:
        return self.__backward

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = remove_price_dependency(df=df)

        df["tr"] = df.high - df.low
        df["roc"] = df.close - df.open

        df["mean-tr"] = df.tr.rolling(window=self.backward).mean()
        df["std-tr"] = df.tr.rolling(window=self.backward).std()
        df["mean-roc"] = df.roc.rolling(window=self.backward).mean()
        df["std-roc"] = df.roc.rolling(window=self.backward, min_periods=1, center=True).std()

        return df

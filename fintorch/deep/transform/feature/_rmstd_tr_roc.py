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
            features=["tr", "roc", "clean-tr", "clean-roc"],
        )
        self.__backward: int = backward

    @property
    def backward(self) -> int:
        return self.__backward

    def transform_df(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = remove_price_dependency(df=df)

        df["mean-open"] = df.open.rolling(window=self.backward, min_periods=1, center=True).mean()
        df["mean-high"] = df.high.rolling(window=self.backward, min_periods=1, center=True).mean()
        df["mean-low"] = df.low.rolling(window=self.backward, min_periods=1, center=True).mean()
        df["mean-close"] = df.close.rolling(window=self.backward, min_periods=1, center=True).mean()

        df["tr"] = df.high - df.low
        df["roc"] = df.close - df.open
        df["clean-tr"] = df["mean-high"] - df["mean-low"]
        df["clean-roc"] = df["mean-close"] - df["mean-open"]

        return df

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df / (df.max() - df.min())

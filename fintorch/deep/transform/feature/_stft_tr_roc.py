import pandas as pd

from ._feature_transform import FeatureTransform
from ....enum import TimeFrame
from ....utils.preprocess import remove_price_dependency, sftf


class StftTrRocFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, muting_percentage: int = 90):
        super().__init__(
            name="Short Term Fourier Transform of True Range and Rate of Change",
            short_name="STFT",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=dim_sequence * 2,
            features=["tr", "roc", "clean-tr", "clean-roc"],
        )

        self.__muting_percentage: int = muting_percentage

    @property
    def muting_percentage(self) -> int:
        return self.__muting_percentage

    def transform_df(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = remove_price_dependency(df=df)

        df["clean-open"] = sftf(array=df.open, muting_percentage=self.muting_percentage)
        df["clean-high"] = sftf(array=df.high, muting_percentage=self.muting_percentage)
        df["clean-low"] = sftf(array=df.low, muting_percentage=self.muting_percentage)
        df["clean-close"] = sftf(array=df.close, muting_percentage=self.muting_percentage)

        df["tr"] = df.high - df.low
        df["roc"] = df.close - df.open
        df["clean-tr"] = df["clean-high"] - df["clean-low"]
        df["clean-roc"] = df["clean-close"] - df["clean-open"]

        return df

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df / (df.max() - df.min())

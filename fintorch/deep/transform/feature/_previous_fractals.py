import pandas as pd

from ._feature_transform import FeatureTransform
from ....enum import TimeFrame
from ....utils.preprocess import remove_price_dependency


class PreviousFractalsFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, length: int = 5):
        super().__init__(
            name="Previous Fractals Feature Transform",
            short_name="PREFC",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=length * dim_sequence * 2,
            features=["open", "high", "low", "close"]
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = remove_price_dependency(df=df)

        df["center-max"] = df.close.rolling(window=self.length, min_periods=1, center=True).max()
        df["center-min"] = df.close.rolling(window=self.length, min_periods=1, center=True).min()
        df["fractal"] = (df.close == df["center-max"]) | (df.close == df["center-min"])
        df = df[df.fractal]
        df = df.dropna()

        return df

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df - df.low.min() / (df.high.max() - df.low.min())

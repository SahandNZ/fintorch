import pandas as pd

from ._feature_transform import FeatureTransform
from ....enum import TimeFrame
from ....utils.preprocess import remove_price_dependency, add_fractals


class PreviousFractalsFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, length: int = 10):
        super().__init__(
            name="Previous Fractals Feature Transform",
            short_name="PREFC",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=length * dim_sequence * 2,
            features=["close", "volume", "trade", "distance"]
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = remove_price_dependency(df=df)
        df = add_fractals(df=df, length=self.length)
        df = df[df["is-fractal"]]
        df["distance"] = df.index.to_series().diff() // self.time_frame
        df = df.dropna()

        return df

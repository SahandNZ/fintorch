import numpy as np
import pandas as pd

from ._feature_transform import FeatureTransform
from ....dtype import DataCollection
from ....enum import TimeFrame
from ....utils.preprocess import remove_price_dependency, add_fractals


class PreviousFractalsWithFundingRateFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, length: int = 5):
        super().__init__(
            name="Previous Fractals With Funding Rate Feature Transform",
            short_name="PREFC2",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=length * dim_sequence * 2,
            features=["close", "rate", "volume", "trade"]
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def _transform_dc_to_sf(self, dc: DataCollection, timestamp: int) -> np.array:
        candles_df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame)
        funding_rates_df = dc.get_funding_rates_df(symbol=self.symbol)

        # merge dataframes
        df = candles_df.copy()
        df["rate"] = funding_rates_df["rate"]
        df["rate"] = df.rate.bfill()

        # backward cropping feature dataframe with timestamp and sequence length
        df = df[df.index < timestamp].iloc[-self.look_back:]
        fdf = self.transform_df(df=df)
        fdf = fdf.iloc[-self.dim_sequence:]
        fdf = fdf[self.features]

        if self.dim_sequence != len(fdf):
            return np.array((self.dim_sequence, self.dim_feature))

        ndf = self.normalize_df(df=fdf)
        sf = ndf.to_numpy()

        return sf

    def transform_df(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = remove_price_dependency(df=df)
        df = add_fractals(df=df, length=self.length)
        df = df[df["is-fractal"]]
        df = df.dropna()

        return df

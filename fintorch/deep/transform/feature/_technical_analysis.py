import numpy as np
import pandas as pd
import ta

from ._feature_transform import FeatureTransform
from ....enum import TimeFrame


class TechnicalAnalysisFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, backward: int = 11):
        super().__init__(
            name="Technical Analysis Feature Transform",
            short_name="TA",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=(backward + dim_sequence) * 2,
            features=[""],
        )
        self.__backward: int = backward

    @property
    def backward(self) -> int:
        return self.__backward

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["roc"] = df.close / df.open - 1
        df["cum-sum-roc"] = np.cumsum(df.roc)

        df["adx"] = ta.trend.adx(df.high, df.low, df.close, window=self.backward, fillna=True)
        df["dpo"] = ta.trend.dpo(df["cum-sum-roc"], window=self.backward, fillna=True)
        df["stc"] = ta.trend.stc(df["cum-sum-roc"], fillna=True)
        df["kst"] = ta.trend.kst(df["cum-sum-roc"], fillna=True)

        return df

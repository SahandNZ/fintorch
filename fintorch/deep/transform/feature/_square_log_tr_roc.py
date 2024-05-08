import numpy as np
import pandas as pd

from ._feature_transform import FeatureTransform
from ....enum import TimeFrame


class SquareLogTrRocFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int):
        super().__init__(
            name="Square Log of True Range and Rate Of Change",
            short_name="SLOG",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_back=dim_sequence * 2,
            features=["tr", "roc", "square-log-tr", "square-log-roc"],
        )

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["square-log-tr"] = np.log(df.high / df.low) ** 2
        df["square-log-roc"] = np.log(df.close / df.open) ** 2

        return df

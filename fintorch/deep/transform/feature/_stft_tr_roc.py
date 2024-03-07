import pandas as pd

from ._feature_transform import FeatureTransform
from ....enum import TimeFrame
from ....utils.signal import DFFT


class StftTrRocFeatureTransform(FeatureTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, muting_percentage: int = 90):
        super().__init__(
            name="Short Term Fourier Transform of True Range and Rate of Change",
            short_name="STFT",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            features=["tr", "roc", "clean-tr", "clean-roc"],
        )
        self.dfft = DFFT(muting_percentage=muting_percentage)

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["clean-tr"] = self.dfft.transform(df.tr)
        df["clean-roc"] = self.dfft.transform(df.roc)
        df = df.dropna()

        return df

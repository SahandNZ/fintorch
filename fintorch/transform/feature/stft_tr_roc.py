from typing import List

import pandas as pd

from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.signal.dfft import DfftTransform


class StftTrRocFeatureTransform(FeatureTransform):
    def __init__(self, symbols: List[str], time_frames: List[int], muting_percentage: int = 95,
                 sequence_length: int = 32):
        super().__init__(
            name="Short Term Fourier Transform of True Range and Rate of Change",
            short_name="STFT",
            description="",
            symbols=symbols,
            time_frames=time_frames,
            features=["tr", "roc", "clean-tr", "clean-roc"],
            sequence_length=sequence_length,
        )
        self.dfft: DfftTransform = DfftTransform(muting_percentage=muting_percentage)

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["clean-tr"] = self.dfft.transform(df.tr)
        df["clean-roc"] = self.dfft.transform(df.roc)
        df = df.dropna()

        return df

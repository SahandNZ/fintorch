import pandas as pd

from fintorch.transform.feature.transform import FeatureTransform
from fintorch.transform.signal.dfft import DfftTransform


class StftTrRocFeatureTransform(FeatureTransform):
    def __init__(self, muting_percentage: int = 95):
        super().__init__(
            name="Short Term Fourier Transform of True Range and Rate of Change",
            short_name="STFT",
            features=["tr", "roc", "clean-tr", "clean-roc"],
            look_back=None
        )
        self.dfft: DfftTransform = DfftTransform(muting_percentage=muting_percentage)

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["clean-tr"] = self.dfft.transform(df.tr)
        df["clean-roc"] = self.dfft.transform(df.roc)
        df = df.dropna()

        return df

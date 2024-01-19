import pandas as pd

from ._feature_transform import FeatureTransform
from ....utils.signal import DFFT


class StftTrRocFeatureTransform(FeatureTransform):
    def __init__(self, sequence_length: int, muting_percentage: int = 95):
        super().__init__(
            name="Short Term Fourier Transform of True Range and Rate of Change",
            short_name="STFT",
            sequence_length=sequence_length,
            look_back=None,
            features=["tr", "roc", "clean-tr", "clean-roc"],
        )
        self.dfft = DFFT(muting_percentage=muting_percentage)

    def _fit_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["clean-tr"] = self.dfft.transform(df.tr)
        df["clean-roc"] = self.dfft.transform(df.roc)
        df = df.dropna()

        return df

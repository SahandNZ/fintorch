import pandas as pd

from fintorch.transform.feature.transform import FeatureTransform


class RollingMeanStdTrRocFeatureTransform(FeatureTransform):
    def __init__(self, look_back: int, sequence_length: int):
        super().__init__(
            name="Rolling Mean and Standard deviation of True Range and Rate Of Change",
            short_name="RMS TR-ROC",
            description="",
            features=["mean-tr", "mean-roc", "std-tr", "std-roc"],
            sequence_length=sequence_length,
        )
        self.__look_back: int = look_back

    @property
    def look_back(self) -> int:
        return self.__look_back

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tr"] = df.high / df.low - 1
        df["roc"] = df.close / df.open - 1
        df["mean-tr"] = df.tr.rolling(self.look_back).mean()
        df["mean-roc"] = df.roc.rolling(self.look_back).mean()
        df["std-tr"] = df.tr.rolling(self.look_back).std()
        df["std-roc"] = df.roc.rolling(self.look_back).std()
        df = df.dropna()

        return df

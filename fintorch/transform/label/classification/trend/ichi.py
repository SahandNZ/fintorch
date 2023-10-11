import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class IchiLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, base_length: int = 5, conversion_length: int = 20):
        super().__init__(
            name="Ichimoku",
            short_name="Ichi",
            description="This labeling method works by comparing base line and conversion line of ichimoku indicator "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            num_classes=2
        )
        self.__base_length: int = base_length
        self.__conversion_length: int = conversion_length

    @property
    def base_length(self) -> int:
        return self.__base_length

    @property
    def conversion_length(self) -> int:
        return self.__conversion_length

    @staticmethod
    def donchian(df: pd.DataFrame, length: int) -> pd.Series:
        return (df.close.rolling(length).max() + df.close.rolling(length).min()) / 2

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["base"] = self.donchian(df, length=self.base_length)
        df["conversion"] = self.donchian(df, length=self.conversion_length)
        df = df.dropna()

        df["up"] = df.conversion < df.base
        df["label"] = df.up.astype(int)

        return df

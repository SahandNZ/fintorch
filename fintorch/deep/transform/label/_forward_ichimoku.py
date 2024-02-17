import pandas as pd

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class ForwardIchimokuLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, base_length: int = 5, conversion_length: int = 20,
                 look_ahead: int = 12):
        super().__init__(
            name="Forward Ichimoku",
            short_name="F-Ichimoku",
            description="This labeling method works by comparing the base line with the conversion line of "
                        "the Ichimoku indicator, and then shifting back the comparison values to assign trend "
                        "labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_ahead=look_ahead,
            classes=["UP", "DOWN"]
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

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["base"] = self.donchian(df, length=self.base_length)
        df["conversion"] = self.donchian(df, length=self.conversion_length)

        df["up"] = df.conversion < df.base
        df["f-up"] = df.up.shift(-self.look_ahead)
        df.dropna(inplace=True)

        df["label"] = df["f-up"].astype(int)

        return df

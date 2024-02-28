import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class ForwardIchimokuLabelTransform(LabelTransform):
    def __init__(
            self,
            symbol: str,
            time_frame: TimeFrame,
            forward: int = 12,
            base_length: int = 5,
            conversion_length: int = 20,
    ):
        super().__init__(
            name="Forward Ichimoku",
            short_name="F-Ichimoku",
            description="This labeling method works by comparing the base line with the conversion line of "
                        "the Ichimoku indicator, and then shifting back the comparison values to assign trend "
                        "labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_back=max(base_length, conversion_length),
            look_ahead=forward,
            classes=["UP", "DOWN"]
        )
        self.__forward: int = forward
        self.__base_length: int = base_length
        self.__conversion_length: int = conversion_length

    @property
    def forward(self) -> int:
        return self.__forward

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

        df["up"] = np.where(np.isnan(df.base) | np.isnan(df.conversion), np.nan, df.conversion < df.base)
        df["f-up"] = df.up.shift(-self.forward)
        df["label"] = np.where(np.isnan(df["f-up"]), np.nan, df["f-up"])

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df.base, label="Base Line")
        ohlc_ax.plot(df.conversion, label="Conversion Line")

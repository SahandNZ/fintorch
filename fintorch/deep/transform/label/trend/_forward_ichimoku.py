import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._label_transform import LabelTransform
from .....enum import TimeFrame


class ForwardIchimokuLabelTransform(LabelTransform):
    def __init__(
            self,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            forward: int = 10,
            base_length: int = 5,
            conversion_length: int = 20,
    ) -> None:
        super().__init__(
            name="Forward Ichimoku",
            short_name="F-Ichimoku",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_ahead=forward,
            look_back=0,
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

    def forward_donchian(self, df: pd.DataFrame, length: int) -> pd.Series:
        forward_highest = df.high.rolling(window=length).max().shift(periods=-self.forward)
        forward_lowest = df.low.rolling(window=length).min().shift(periods=-self.forward)
        forward_middle = (forward_highest + forward_lowest) / 2

        return forward_middle

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["forward-base"] = self.forward_donchian(df=df, length=self.base_length)
        df["forward-conversion"] = self.forward_donchian(df=df, length=self.conversion_length)
        df["up"] = df["forward-conversion"] <= df["forward-base"]
        df["label"] = np.where(np.isnan(df["forward-conversion"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["forward-base"], label="Forward Base Line")
        ohlc_ax.plot(df["forward-conversion"], label="Forward Conversion Line")

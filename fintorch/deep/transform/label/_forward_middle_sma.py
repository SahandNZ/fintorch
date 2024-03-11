import math

import pandas as pd
from matplotlib import pyplot as plt

from ._label_transform import LabelTransform
from ....enum import TimeFrame


class ForwardMiddleSmaLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, forward: int = 5, length: int = 11):
        super().__init__(
            name="Forward Middle Simple Moving Average",
            short_name="F.M-SMA",
            description="This labeling method works by smoothing the close price using a lookahead and "
                        "the Simple Moving Average (SMA) method. It then compares the current smoothed close price "
                        "with the forward values to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=math.ceil(forward + length / 2),
            classes=["UP", "DOWN"]
        )

        self.__length: int = length
        self.__forward: int = forward

    @property
    def length(self) -> int:
        return self.__length

    @property
    def forward(self) -> int:
        return self.__forward

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["middle-sma"] = df.close.rolling(window=self.length, min_periods=1, center=True).mean()
        df["forward-middle-sma"] = df["middle-sma"].shift(periods=-self.forward).ffill()
        df["label"] = df["middle-sma"] <= df["forward-middle-sma"]

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["middle-sma"], label="Middle SMA")
        ohlc_ax.plot(df["forward-middle-sma"], label="Forward Middle SMA")

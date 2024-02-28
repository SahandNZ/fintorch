import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class ForwardMiddleSmaLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, backward: int = 51, forward: int = 12):
        super().__init__(
            name="Forward Middle Simple Moving Average",
            short_name="F.M-SMA",
            description="This labeling method works by smoothing the close price using a lookahead and "
                        "the Simple Moving Average (SMA) method. It then compares the current smoothed close price "
                        "with the forward values to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_back=backward // 2 + 1,
            look_ahead=backward // 2 + 1 + forward,
            classes=["UP", "DOWN"]
        )

        self.__backward: int = backward
        self.__forward: int = forward

    @property
    def backward(self) -> int:
        return self.__backward

    @property
    def forward(self) -> int:
        return self.__forward

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["msma"] = df.close.rolling(self.backward, center=True).mean()
        df["f-msma"] = df.msma.shift(-self.forward)

        df["up"] = df.msma < df["f-msma"]
        df["label"] = np.where(np.isnan(df.msma) | np.isnan(df["f-msma"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df.msma, label="Middle SMA")
        ohlc_ax.plot(df["f-msma"], label="Forward Middle SMA")

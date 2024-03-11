import pandas as pd
from matplotlib import pyplot as plt

from ._label_transform import LabelTransform
from ....enum import TimeFrame


class TripleBarrierLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, forward: int = 10):
        super().__init__(
            name="Triple Barrier",
            short_name="T-Barrier",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=forward,
            classes=["UP", "DOWN"]
        )

        self.__forward: int = forward

    @property
    def forward(self) -> int:
        return self.__forward

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["forward-close"] = df.close.shift(periods=-self.forward).ffill()
        df["label"] = df.close <= df["forward-close"]

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

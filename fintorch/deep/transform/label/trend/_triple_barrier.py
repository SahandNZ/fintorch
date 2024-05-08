import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fintorch.deep.transform.label._label_transform import LabelTransform
from fintorch.enum import TimeFrame


class TripleBarrierLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, forward: int = 10) -> None:
        super().__init__(
            name="Triple Barrier",
            short_name="T-Barrier",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_ahead=forward,
            look_back=0,
            classes=["UP", "DOWN"]
        )

        self.__forward: int = forward

    @property
    def forward(self) -> int:
        return self.__forward

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["forward-close"] = df.close.shift(periods=-self.forward)
        df["up"] = df.close <= df["forward-close"]
        df["label"] = np.where(np.isnan(df["forward-close"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

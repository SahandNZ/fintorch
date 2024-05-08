import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._label_transform import LabelTransform
from .....enum import TimeFrame


class ForwardRocLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, forward: int = 10) -> None:
        super().__init__(
            name="Forward Rate Of Change",
            short_name="F-ROC",
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
        df["roc"] = df.close / df.open - 1
        df["sum-roc"] = df.roc.rolling(window=self.forward).sum()
        df["forward-sum-roc"] = df["sum-roc"].shift(periods=-self.forward)
        df["up"] = 0 <= df["forward-sum-roc"]
        df["label"] = np.where(np.isnan(df["forward-sum-roc"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

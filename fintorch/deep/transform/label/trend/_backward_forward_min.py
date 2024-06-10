import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._backward_forward import BackwardForwardLabelTransform
from .....enum import TimeFrame


class BackwardForwardMinimumLabelTransform(BackwardForwardLabelTransform):
    def __init__(
            self,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            backward: int = 20,
            forward: int = 20
    ) -> None:
        super().__init__(
            name="Backward Forward min",
            short_name="B.F-Min",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            backward=backward,
            forward=forward,
            classes=["UP", "DOWN"]
        )

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["backward-min"] = df.close.rolling(window=self.backward).min()
        df["forward-min"] = df.close.rolling(window=self.forward).min().shift(periods=-self.forward)
        df["up"] = df["backward-min"] <= df["forward-min"]
        df["label"] = np.where(np.isnan(df["forward-min"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["backward-min"], label="Backward Min")
        ohlc_ax.plot(df["forward-min"], label="Forward Min")

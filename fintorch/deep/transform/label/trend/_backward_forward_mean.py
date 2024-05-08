import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._backward_forward import BackwardForwardLabelTransform
from .....enum import TimeFrame


class BackwardForwardMeanLabelTransform(BackwardForwardLabelTransform):
    def __init__(
            self,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            backward: int = 10,
            forward: int = 10
    ) -> None:
        super().__init__(
            name="Backward Forward Mean",
            short_name="B.F-MEAN",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            backward=backward,
            forward=forward,
            classes=["UP", "DOWN"]
        )

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.backward + self.forward
        df["backward-forward-mean"] = df.close.rolling(window=window).mean().shift(-self.forward)
        df["up"] = df["backward-forward-mean"].shift(1) <= df["backward-forward-mean"]
        df["label"] = np.where(np.isnan(df["backward-forward-mean"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["backward-forward-mean"], label="backward Forward Mean")

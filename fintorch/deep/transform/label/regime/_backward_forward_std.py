import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._backward_forward import BackwardForwardLabelTransform
from .....enum import TimeFrame


class BackwardForwardStdLabelTransform(BackwardForwardLabelTransform):
    def __init__(
            self,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            backward: int = 10,
            forward: int = 10
    ) -> None:
        super().__init__(
            name="Backward Forward Std",
            short_name="B-F.STD",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            backward=backward,
            forward=forward,
            classes=["Range", "Trend"]
        )

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["backward-forward-std"] = df.close.rolling(window=self.backward + self.forward).std().shift(-self.forward)
        df["trend"] = df["backward-forward-std"].mean() <= df["backward-forward-std"]
        df["label"] = np.where(np.isnan(df["backward-forward-std"]), np.nan, df.trend)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

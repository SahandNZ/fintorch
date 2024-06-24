import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._backward_forward import BackwardForwardLabelTransform
from .....enum import TimeFrame


class BackwardForwardAverageLabelTransform(BackwardForwardLabelTransform):
    def __init__(
            self,
            symbol: str,
            time_frame: TimeFrame,
            dim_sequence: int,
            backward: int = 20,
            forward: int = 20
    ) -> None:
        super().__init__(
            name="Backward Forward Average",
            short_name="B.F-Avg",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            backward=backward,
            forward=forward,
            classes=["UP", "DOWN"],
            data_loader_kwargs={"batch_size": None, "batch_count": 2 ** 3},
            optimizer_kwargs={"lr": 1e-5},
            lr_scheduler_kwargs={},
            trainer_kwargs={"epochs_count": 50}
        )

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.backward + self.forward
        df["backward-forward-avg"] = df.close.rolling(window=window).mean().shift(-self.forward)
        df["up"] = df["backward-forward-avg"].shift(1) <= df["backward-forward-avg"]
        df["label"] = np.where(np.isnan(df["backward-forward-avg"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["backward-forward-avg"], label="Backward forward average")

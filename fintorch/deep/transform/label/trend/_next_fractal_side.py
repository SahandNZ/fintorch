import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._label_transform import LabelTransform
from .....enum import TimeFrame
from .....utils.preprocess import add_fractals


class NextFractalSideLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, length: int = 21) -> None:
        super().__init__(
            name="Next Fractal Side",
            short_name="N-Fractal-Side",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_ahead=length * 2,
            look_back=length * 2,
            classes=["UP", "DOWN"]
        )

        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_fractals(df=df, length=self.length)
        df["up"] = np.where(df["is-fractal"], df["is-high-fractal"], np.nan)
        df["label"] = df.up.bfill()

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["fractal"], "y", label="Fractal")

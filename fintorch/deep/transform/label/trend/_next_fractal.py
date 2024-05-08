import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .._label_transform import LabelTransform
from .....enum import TimeFrame
from .....utils.preprocess import add_fractals


class NextFractalLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int, length: int = 21) -> None:
        super().__init__(
            name="Next Fractal",
            short_name="N-Fractal",
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
        df["next-fractal"] = df.fractal.shift(-1).bfill()
        df["up"] = df.close <= df["next-fractal"]
        df["label"] = np.where(np.isnan(df["next-fractal"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["next-fractal"], "y", label="Next Fractal")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class NextFractalLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, length: int = 21):
        super().__init__(
            name="Next Fractal",
            short_name="N-Fractal",
            description="This labeling method works by comparing the next fractal with the current close price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_back=length // 2 * 10,
            look_ahead=length // 2 * 10,
            classes=["UP", "DOWN"]
        )

        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["cmin"] = df.close.rolling(self.length, center=True).min()
        df["cmax"] = df.close.rolling(self.length, center=True).max()

        df['isfl'] = df.close == df.cmin
        df["isfh"] = df.close == df.cmax
        df["fractal"] = df.close[df.isfl | df.isfh]
        df["next-fractal"] = df.fractal.shift(-1).bfill()

        df["up"] = df.close <= df["next-fractal"]
        df["label"] = np.where(np.isnan(df["next-fractal"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["next-fractal"], "yo", label="Next Fractal")

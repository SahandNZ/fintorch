import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fintorch.plot import draw_trend_rectangles
from fintorch.transform.label.transform import LabelTransform


class NextFractalLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, length: int = 10):
        super().__init__(
            name="Forward Fractal",
            short_name="F-Fractal",
            description="This labeling method works by comparing the next fractal with the current close price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            num_classes=2
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bmin"] = df.low.rolling(self.length).min()
        df["bmax"] = df.high.rolling(self.length).max()
        df["fmin"] = df.low.rolling(self.length).min().shift(-self.length + 1)
        df["fmax"] = df.high.rolling(self.length).max().shift(-self.length + 1)

        df['isfl'] = (df.low == df.fmin) & (df.low == df.bmin)
        df["isfh"] = (df.high == df.fmax) & (df.high == df.bmax)
        df["fractal"] = np.where(df.isfl, df.low, np.where(df.isfh, df.high, np.nan))
        df["next-fractal"] = df.fractal.shift(-1).bfill()
        df.drop(columns=["fractal"], inplace=True)
        df.dropna(inplace=True)

        df["up"] = df.close <= df["next-fractal"]
        df["label"] = df.up.astype(int)

        return df

    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        nf = df["next-fractal"]
        nf_lines = np.where(nf == nf.shift(1), nf, np.nan)
        ohlcv_ax.plot(nf_lines, label="Next Fractal")
        draw_trend_rectangles(ax=ohlcv_ax, df=df)
        ohlcv_ax.legend()

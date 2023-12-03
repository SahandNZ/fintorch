import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.plot import draw_trend


class NextFractalLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, look_ahead: int = 5, look_back: int = 5):
        super().__init__(
            name="Next Fractal",
            short_name="N-Fractal",
            description="This labeling method works by comparing the next fractal with the current close price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=look_ahead,
            num_classes=2
        )
        self.__look_back: int = look_back

    @property
    def look_back(self) -> int:
        return self.__look_back

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bmin"] = df.low.rolling(self.look_back).min()
        df["bmax"] = df.high.rolling(self.look_back).max()
        df["fmin"] = df.low.rolling(self.look_ahead).min().shift(-self.look_ahead + 1)
        df["fmax"] = df.high.rolling(self.look_ahead).max().shift(-self.look_ahead + 1)

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
        draw_trend(ax=ohlcv_ax, df=df)
        ohlcv_ax.legend()

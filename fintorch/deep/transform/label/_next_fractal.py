import pandas as pd
from matplotlib import pyplot as plt

from ._label_transform import LabelTransform
from ....enum import TimeFrame


class NextFractalLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, length: int = 11):
        super().__init__(
            name="Next Fractal",
            short_name="N-Fractal",
            description="This labeling method works by comparing the next fractal with the current close price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=length * 2,
            classes=["UP", "DOWN"]
        )

        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["center-max"] = df.close.rolling(window=self.length, min_periods=1, center=True).max()
        df["center-min"] = df.close.rolling(window=self.length, min_periods=1, center=True).min()
        df["is-high-fractal"] = df.close == df["center-max"]
        df["is-low-fractal"] = df.close == df["center-min"]
        df["fractal"] = df.close[df["is-high-fractal"] | df["is-low-fractal"]]
        df["next-fractal"] = df.fractal.shift(-1).bfill().ffill()
        df["label"] = df.close <= df["next-fractal"]

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["next-fractal"], "y", label="Next Fractal")

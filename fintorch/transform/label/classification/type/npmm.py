import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fintorch.transform.label.transform import LabelTransform


class NpmmLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, look_ahead: int = 10):
        super().__init__(
            name="N Period Min Max",
            short_name="NPMM",
            description="This labeling method works by calculating the N-Period Min-Max indicator "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=look_ahead,
            num_classes=3
        )

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bmin"] = df.close.rolling(self.look_ahead).min()
        df["bmax"] = df.close.rolling(self.look_ahead).max()
        df["fmin"] = df.close.rolling(self.look_ahead).min().shift(-self.look_ahead + 1)
        df["fmax"] = df.close.rolling(self.look_ahead).max().shift(-self.look_ahead + 1)

        df["ispl"] = (df.close == df.fmin) & (df.close == df.bmin)
        df["isph"] = (df.close == df.fmax) & (df.close == df.bmax)
        df["label"] = np.where(df.isph, 0, np.where(df.ispl, 1, 2))
        df = df.dropna()

        return df

    # TODO
    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        pass

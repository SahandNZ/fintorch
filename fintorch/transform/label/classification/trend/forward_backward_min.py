import pandas as pd
from matplotlib import pyplot as plt

from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.plot import draw_trend


class ForwardBackwardMinimumLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, look_ahead: int = 10, look_back: int = 10, ):
        super().__init__(
            name="Forward Backward Min",
            short_name="F.B-Min",
            description="This labeling method works by comparing the Backward Min series with "
                        "the Forward Min series to assign trend labels to the data.",
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
        df["bmin"] = df.close.rolling(self.look_back).min()
        df["fmin"] = df.close.rolling(self.look_ahead).min().shift(-self.look_ahead + 1)
        df.dropna(inplace=True)

        df["up"] = df.bmin <= df.fmin
        df["label"] = df.up.astype(int)

        return df

    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        ohlcv_ax.plot(df.bmin, label="Backward Min")
        ohlcv_ax.plot(df.fmin, label="Forward Min")
        draw_trend(ax=ohlcv_ax, df=df)
        ohlcv_ax.legend()

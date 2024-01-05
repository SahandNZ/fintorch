import pandas as pd
from matplotlib import pyplot as plt

from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.plot import draw_trend


class ForwardMiddleSmaLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, look_ahead: int = 12, length: int = 51):
        super().__init__(
            name="Forward Middle Simple Moving Average",
            short_name="F.M-SMA",
            description="This labeling method works by smoothing the close price using a lookahead and "
                        "the Simple Moving Average (SMA) method. It then compares the current smoothed close price "
                        "with the forward values to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=look_ahead,
            num_classes=2
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["msma"] = df.close.rolling(self.length).mean().shift(-self.length // 2)
        df["f-msma"] = df.msma.shift(-self.look_ahead)
        df.dropna(inplace=True)

        df["up"] = df.msma < df["f-msma"]
        df["label"] = df.up.astype(int)

        return df

    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        ohlcv_ax.plot(df.msma, label="Middle SMA")
        draw_trend(ax=ohlcv_ax, df=df)
        ohlcv_ax.legend()

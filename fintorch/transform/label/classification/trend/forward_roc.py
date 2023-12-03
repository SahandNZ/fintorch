import pandas as pd
from matplotlib import pyplot as plt

from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.plot import draw_trend


class ForwardRocLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, look_ahead: int = 12):
        super().__init__(
            name="Forward Rate Of Change",
            short_name="F-ROC",
            description="This labeling method works by calculating the sum of the rate of change values. "
                        'If the sum value is positive, it assigns an "up trend" label to the data. Conversely, '
                        'if the sum value is negative, it assigns a "down trend" label to the data.',
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=look_ahead,
            num_classes=2
        )

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["roc"] = df.close / df.open - 1
        df["f-roc"] = df.roc.rolling(self.look_ahead).sum().shift(-self.look_ahead)
        df.dropna(inplace=True)

        df["up"] = 0 < df["f-roc"]
        df["label"] = df.up.astype(int)

        return df

    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        draw_trend(ax=ohlcv_ax, df=df)
        ohlcv_ax.legend()

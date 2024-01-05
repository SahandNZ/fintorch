import pandas as pd
from matplotlib import pyplot as plt

from fintorch.transform.label.transform import LabelTransform
from fintorch.utils.plot import draw_trend


class UpDownLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int):
        super().__init__(
            name="Up-Down",
            short_name="Up-Down",
            description="This labeling method compares the current close price with the current open price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=1,
            num_classes=2
        )

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["up"] = df.close < df.close.shift(-self.look_ahead)
        df["label"] = df.up.astype(int)
        df.dropna(inplace=True)

        return df

    def _draw_lines(self, df: pd.DataFrame, ohlcv_ax: plt.Axes, volume_ax: plt.Axes) -> pd.DataFrame:
        draw_trend(ax=ohlcv_ax, df=df)
        ohlcv_ax.legend()

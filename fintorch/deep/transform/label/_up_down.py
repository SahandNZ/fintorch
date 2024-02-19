import pandas as pd
from matplotlib import pyplot as plt

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class UpDownLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame):
        super().__init__(
            name="Up-Down",
            short_name="Up-Down",
            description="This labeling method compares the current close price with the current open price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_back=0,
            look_ahead=0,
            classes=["UP", "DOWN"]
        )

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["up"] = df.open < df.close
        df["label"] = df.up.astype(int)
        df.dropna(inplace=True)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

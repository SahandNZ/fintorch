import pandas as pd
from matplotlib import pyplot as plt

from .._label_transform import LabelTransform
from .....enum import TimeFrame


class UpDownLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int) -> None:
        super().__init__(
            name="Up-Down",
            short_name="Up-Down",
            description="This labeling method compares the current close price with the current open price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_ahead=1,
            look_back=0,
            classes=["UP", "DOWN"]
        )

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["label"] = df.open <= df.close

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

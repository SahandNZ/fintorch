import pandas as pd
from matplotlib import pyplot as plt

from .._label_transform import LabelTransform
from .....enum import TimeFrame
from .....utils.preprocess import add_heikin_ashi


class UpDownHeikinAshiLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, dim_sequence: int) -> None:
        super().__init__(
            name="Up Down Heikin Ashi",
            short_name="F-HA",
            description="",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=dim_sequence,
            look_ahead=1,
            look_back=0,
            classes=["UP", "DOWN"]
        )

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_heikin_ashi(df=df)
        df["ha-color"] = df["ha-open"] < df["ha-close"]
        df["label"] = df["ha-color"]

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

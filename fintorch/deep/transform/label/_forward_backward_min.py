import pandas as pd
from matplotlib import pyplot as plt

from ._label_transform import LabelTransform
from ....enum import TimeFrame


class ForwardBackwardMinimumLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, backward: int = 10, forward: int = 10):
        super().__init__(
            name="Forward Backward Min",
            short_name="F.B-Min",
            description="This labeling method works by comparing the Backward Min series with "
                        "the Forward Min series to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            look_ahead=forward,
            classes=["UP", "DOWN"]
        )

        self.__backward: int = backward
        self.__forward: int = forward

    @property
    def backward(self) -> int:
        return self.__backward

    @property
    def forward(self) -> int:
        return self.__forward

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        forward_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.forward)
        df["backward-min"] = df.close.rolling(window=self.backward, min_periods=1).min()
        df["forward-min"] = df.close.rolling(window=forward_indexer, min_periods=1).min()
        df["label"] = df["backward-min"] <= df["forward-min"]

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        ohlc_ax.plot(df["backward-min"], label="Backward Min")
        ohlc_ax.plot(df["forward-min"], label="Forward Min")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class ForwardRocLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, forward: int = 12):
        super().__init__(
            name="Forward Rate Of Change",
            short_name="F-ROC",
            description="This labeling method works by calculating the sum of the rate of change values. "
                        'If the sum value is positive, it assigns an "up trend" label to the data. Conversely, '
                        'if the sum value is negative, it assigns a "down trend" label to the data.',
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_back=0,
            look_ahead=forward,
            classes=["UP", "DOWN"]
        )

        self.__forward: int = forward

    @property
    def forward(self) -> int:
        return self.__forward

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["roc"] = df.close / df.open - 1
        df["f-roc"] = df.roc.rolling(self.forward).sum().shift(-self.forward + 1)

        df["up"] = 0 < df["f-roc"]
        df["label"] = np.where(np.isnan(df["f-roc"]), np.nan, df.up)

        return df

    def _draw_lines(self, ohlc_ax: plt.Axes, df: pd.DataFrame) -> None:
        pass

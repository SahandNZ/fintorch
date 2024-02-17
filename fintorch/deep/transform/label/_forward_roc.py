import pandas as pd

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class ForwardRocLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, look_ahead: int = 12):
        super().__init__(
            name="Forward Rate Of Change",
            short_name="F-ROC",
            description="This labeling method works by calculating the sum of the rate of change values. "
                        'If the sum value is positive, it assigns an "up trend" label to the data. Conversely, '
                        'if the sum value is negative, it assigns a "down trend" label to the data.',
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_ahead=look_ahead,
            classes=["UP", "DOWN"]
        )

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["roc"] = df.close / df.open - 1
        df["f-roc"] = df.roc.rolling(self.look_ahead).sum().shift(-self.look_ahead)
        df.dropna(inplace=True)

        df["up"] = 0 < df["f-roc"]
        df["label"] = df.up.astype(int)

        return df

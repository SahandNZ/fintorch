import pandas as pd

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class ForwardMiddleSmaLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, length: int = 51, forward: int = 12):
        super().__init__(
            name="Forward Middle Simple Moving Average",
            short_name="F.M-SMA",
            description="This labeling method works by smoothing the close price using a lookahead and "
                        "the Simple Moving Average (SMA) method. It then compares the current smoothed close price "
                        "with the forward values to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            forward=forward,
            classes=["UP", "DOWN"]
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["msma"] = df.close.rolling(self.length).mean().shift(-self.length // 2)
        df["f-msma"] = df.msma.shift(-self.forward)
        df.dropna(inplace=True)

        df["up"] = df.msma < df["f-msma"]
        df["label"] = df.up.astype(int)

        return df

import pandas as pd

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class ForwardBackwardMinimumLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, forward: int = 10, backward: int = 10):
        super().__init__(
            name="Forward Backward Min",
            short_name="F.B-Min",
            description="This labeling method works by comparing the Backward Min series with "
                        "the Forward Min series to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_back=backward,
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
        df["bmin"] = df.close.rolling(self.backward).min()
        df["fmin"] = df.close.rolling(self.forward).min().shift(-self.forward + 1)
        df.dropna(inplace=True)

        df["up"] = df.bmin <= df.fmin
        df["label"] = df.up.astype(int)

        return df

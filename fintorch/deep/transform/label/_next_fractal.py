import numpy as np
import pandas as pd

from ....enum import TimeFrame
from ._label_transform import LabelTransform


class NextFractalLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: TimeFrame, backward: int = 10, forward: int = 10):
        super().__init__(
            name="Next Fractal",
            short_name="N-Fractal",
            description="This labeling method works by comparing the next fractal with the current close price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            dim_sequence=1,
            look_back=backward * 10,
            look_ahead=forward * 10,
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
        df["bmin"] = df.low.rolling(self.backward).min()
        df["bmax"] = df.high.rolling(self.backward).max()
        df["fmin"] = df.low.rolling(self.forward).min().shift(-self.forward + 1)
        df["fmax"] = df.high.rolling(self.forward).max().shift(-self.forward + 1)

        df['isfl'] = (df.low == df.fmin) & (df.low == df.bmin)
        df["isfh"] = (df.high == df.fmax) & (df.high == df.bmax)
        df["fractal"] = np.where(df.isfl, df.low, np.where(df.isfh, df.high, np.nan))
        df["next-fractal"] = df.fractal.shift(-1).bfill()
        df.drop(columns=["fractal"], inplace=True)
        df.dropna(inplace=True)

        df["up"] = df.close <= df["next-fractal"]
        df["label"] = df.up.astype(int)

        return df

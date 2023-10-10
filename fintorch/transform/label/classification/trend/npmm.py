import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class NpmmLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, length: int = 10):
        super().__init__(
            name="N Period Min Max",
            short_name="NPMM",
            description="This labeling method works by calculating the N-Period Min-Max indicator "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            label2side={0: -1, 1: 0, 2: 1}
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bmin"] = df.close.rolling(self.length).min()
        df["bmax"] = df.close.rolling(self.length).max()
        df["fmin"] = df.close.rolling(self.length).min().shift(-self.length + 1)
        df["fmax"] = df.close.rolling(self.length).max().shift(-self.length + 1)

        df["ispl"] = (df.close == df.fmin) & (df.close == df.bmin)
        df["isph"] = (df.close == df.fmax) & (df.close == df.bmax)
        df["label"] = np.where(df.isph, 0, np.where(df.ispl, 2, 1))
        df = df.dropna()

        return df

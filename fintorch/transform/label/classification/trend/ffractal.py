import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class FFractalLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, length: int = 10):
        super().__init__(
            name="Forward Fractal",
            short_name="F-Fractal",
            description="This labeling method works by comparing the next fractal with the current close price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            label2side={0: -1, 1: 1}
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
        df["isp"] = df.ispl | df.isph

        df["p"] = df.close[df.isp]
        df["f-pivot"] = df.p.shift(-1).bfill()
        df["up"] = df.close < df["f-pivot"]
        df["label"] = np.where(df.up, 1, 0)
        df = df.dropna()

        return df

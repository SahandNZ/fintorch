import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class BfmmLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, look_back: int = 10, look_ahead: int = 10):
        super().__init__(
            name="Backward Forward Min Max",
            short_name="BFMM",
            description="This labeling method works by comparing the Backward Min-Max series with "
                        "the Forward Min-Max series to assign trend labels to the data.Additionally, "
                        "this method can detect sideways markets and assign a label value of 0 to them.",
            symbol=symbol,
            time_frame=time_frame,
            num_classes=3
        )
        self.__look_back: int = look_back
        self.__look_ahead: int = look_ahead

    @property
    def look_back(self) -> int:
        return self.__look_back

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bmin"] = df.close.rolling(self.look_back).min()
        df["bmax"] = df.close.rolling(self.look_back).max()
        df["fmin"] = df.close.rolling(self.look_ahead).min().shift(-self.look_ahead + 1)
        df["fmax"] = df.close.rolling(self.look_ahead).max().shift(-self.look_ahead + 1)
        df = df.dropna()

        df["down"] = df.fmax <= df.bmax
        df["up"] = df.bmin <= df.fmin
        df["label"] = np.where(df.down, 0, np.where(df.up, 1, 2))

        return df

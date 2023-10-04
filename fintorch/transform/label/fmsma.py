import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class FMsmaLabelTransform(LabelTransform):
    def __init__(self, length: int = 51, look_ahead: int = 12):
        super().__init__(
            name="Forwad Middle Simple Moving Average",
            short_name="F-MSMA",
            description="This labeling method works by smoothing the close price using a lookahead and "
                        "the Simple Moving Average (SMA) method. It then compares the current smoothed close price "
                        "with the forward values to assign trend labels to the data.",
            num_classes=3,
            min_class=-1,
        )
        self.__length: int = length
        self.__look_ahead: int = look_ahead

    @property
    def length(self) -> int:
        return self.__length

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["msma"] = df.close.rolling(self.length).mean().shift(-self.length // 2)
        df["up"] = df.msma < df.msma.shift(-self.look_ahead)
        df["label"] = np.where(df.up, 1, -1)
        df = df.dropna()

        return df

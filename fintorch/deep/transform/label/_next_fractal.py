import numpy as np
import pandas as pd

from ._label_transform import LabelTransform


class NextFractalLabelTransform(LabelTransform):
    def __init__(self, look_back: int = 10, look_ahead: int = 10):
        super().__init__(
            name="Next Fractal",
            short_name="N-Fractal",
            description="This labeling method works by comparing the next fractal with the current close price "
                        "to assign trend labels to the data.",
            sequence_length=1,
            look_ahead=look_ahead,
            classes=["UP", "DOWN"]
        )
        self.__look_back: int = look_back

    @property
    def look_back(self) -> int:
        return self.__look_back

    def _fit_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bmin"] = df.low.rolling(self.look_back).min()
        df["bmax"] = df.high.rolling(self.look_back).max()
        df["fmin"] = df.low.rolling(self.look_ahead).min().shift(-self.look_ahead + 1)
        df["fmax"] = df.high.rolling(self.look_ahead).max().shift(-self.look_ahead + 1)

        df['isfl'] = (df.low == df.fmin) & (df.low == df.bmin)
        df["isfh"] = (df.high == df.fmax) & (df.high == df.bmax)
        df["fractal"] = np.where(df.isfl, df.low, np.where(df.isfh, df.high, np.nan))
        df["next-fractal"] = df.fractal.shift(-1).bfill()
        df.drop(columns=["fractal"], inplace=True)
        df.dropna(inplace=True)

        df["up"] = df.close <= df["next-fractal"]
        df["label"] = df.up.astype(int)

        return df

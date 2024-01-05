import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class ForwardBackwardMinimumLabelTransform(LabelTransform):
    def __init__(self, look_ahead: int = 10, look_back: int = 10):
        super().__init__(
            name="Forward Backward Min",
            short_name="F.B-Min",
            description="This labeling method works by comparing the Backward Min series with "
                        "the Forward Min series to assign trend labels to the data.",
            look_ahead=look_ahead,
            classes=["UP", "DOWN"]
        )
        self.__look_back: int = look_back

    @property
    def look_back(self) -> int:
        return self.__look_back

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bmin"] = df.close.rolling(self.look_back).min()
        df["fmin"] = df.close.rolling(self.look_ahead).min().shift(-self.look_ahead + 1)
        df.dropna(inplace=True)

        df["up"] = df.bmin <= df.fmin
        df["label"] = df.up.astype(int)

        return df

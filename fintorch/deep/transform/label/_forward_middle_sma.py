import pandas as pd

from ._label_transform import LabelTransform


class ForwardMiddleSmaLabelTransform(LabelTransform):
    def __init__(self, length: int = 51, look_ahead: int = 12):
        super().__init__(
            name="Forward Middle Simple Moving Average",
            short_name="F.M-SMA",
            description="This labeling method works by smoothing the close price using a lookahead and "
                        "the Simple Moving Average (SMA) method. It then compares the current smoothed close price "
                        "with the forward values to assign trend labels to the data.",
            sequence_length=1,
            look_ahead=look_ahead,
            classes=["UP", "DOWN"]
        )
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["msma"] = df.close.rolling(self.length).mean().shift(-self.length // 2)
        df["f-msma"] = df.msma.shift(-self.look_ahead)
        df.dropna(inplace=True)

        df["up"] = df.msma < df["f-msma"]
        df["label"] = df.up.astype(int)

        return df

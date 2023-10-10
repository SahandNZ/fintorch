import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class FMsmaLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int, length: int = 51, look_ahead: int = 12):
        super().__init__(
            name="Forwad Middle Simple Moving Average",
            short_name="F-MSMA",
            description="This labeling method works by smoothing the close price using a lookahead and "
                        "the Simple Moving Average (SMA) method. It then compares the current smoothed close price "
                        "with the forward values to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            label2side={0: -1, 1: 1},
        )
        self.__length: int = length
        self.__look_ahead: int = look_ahead

    @property
    def length(self) -> int:
        return self.__length

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["msma"] = df.close.rolling(self.length).mean().shift(-self.length // 2)
        df["up"] = df.msma < df.msma.shift(-self.look_ahead)
        df["label"] = df.up.astype(int)
        df = df.dropna()

        return df

import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class UpDownLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int):
        super().__init__(
            name="Up-Down",
            short_name="Up-Down",
            description="This labeling method compares the current close price with the current open price "
                        "to assign trend labels to the data.",
            symbol=symbol,
            time_frame=time_frame,
            num_classes=2
        )

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["up"] = df.open < df.close
        df["label"] = df.up.astype(int)
        df = df.dropna()

        return df

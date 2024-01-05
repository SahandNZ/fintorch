import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class UpDownLabelTransform(LabelTransform):
    def __init__(self):
        super().__init__(
            name="Up-Down",
            short_name="Up-Down",
            description="This labeling method compares the current close price with the current open price "
                        "to assign trend labels to the data.",
            look_ahead=1,
            classes=["UP", "DOWN"]
        )

    def _fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["up"] = df.close < df.close.shift(-self.look_ahead)
        df["label"] = df.up.astype(int)
        df.dropna(inplace=True)

        return df

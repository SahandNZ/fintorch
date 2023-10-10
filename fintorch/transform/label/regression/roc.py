import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class RocLabelTransform(LabelTransform):
    def __init__(self, symbol: str, time_frame: int):
        super().__init__(
            name="Rate of Change",
            short_name="ROC",
            description="",
            symbol=symbol,
            time_frame=time_frame
        )

    def _fit(self, df: pd.DataFrame):
        df["roc"] = df.close / df.open - 1
        df["label"] = df.roc / df.roc.std()
        df = df.dropna()

        return df

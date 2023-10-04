import numpy as np
import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class FrocLabelTransform(LabelTransform):
    def __init__(self, look_ahead: int = 12):
        super().__init__(
            name="Forward Rate Of Change",
            short_name="F-ROC",
            description="This labeling method works by calculating the sum of the rate of change values. "
                        'If the sum value is positive, it assigns an "up trend" label to the data. Conversely, '
                        'if the sum value is negative, it assigns a "down trend" label to the data.',
            num_classes=3,
            min_class=-1,
        )
        self.__look_ahead: int = look_ahead

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    def fit(self, df: pd.DataFrame):
        df["roc"] = df.close / df.open - 1
        df["f-roc"] = df.roc.rolling(self.look_ahead).sum().shift(-self.look_ahead)
        df["up"] = 0 < df["f-roc"]
        df["label"] = np.where(df.up, 1, -1)
        df = df.dropna()

        return df

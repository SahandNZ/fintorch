import pandas as pd

from fintorch.transform.label.transform import LabelTransform


class FMsmaLabelTransform(LabelTransform):
    def __init__(self, length: int = 51, look_ahead: int = 12):
        super().__init__(name="Forward MSMA", num_classes=2)
        self.__length: int = length
        self.__look_ahead: int = look_ahead

    @property
    def length(self) -> int:
        return self.__length

    @property
    def look_ahead(self) -> int:
        return self.__look_ahead

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df['msma'] = df.close.rolling(self.length).mean().shift(-self.length // 2)
        df['label'] = df.msma < df.msma.shift(-self.look_ahead)
        df['label'] = df.label.astype(int)
        df = df.dropna()

        return df

from datetime import datetime
from typing import List, Union

import pandas as pd


class LongShortRatio:
    def __init__(self):
        self.timestamp: Union[int, None] = None
        self.ratio: Union[float, None] = None
        self.long: Union[float, None] = None
        self.short: Union[float, None] = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp) if self.timestamp is not None else None

    @staticmethod
    def to_dataframe(long_short_ratios: List) -> pd.DataFrame:
        data = [LongShortRatio.to_list(long_short_ratio) for long_short_ratio in long_short_ratios]
        columns = ['timestamp', 'ratio', 'long', 'short']
        df = pd.DataFrame(data=data, columns=columns)
        df = df.set_index('timestamp')

        return df

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def from_list(data: List):
        instance = LongShortRatio()

        instance.timestamp = int(data[0])
        instance.ratio = float(data[1])
        instance.long = float(data[2])
        instance.short = float(data[3])

        return instance

    def to_list(self):
        return [self.timestamp, self.ratio, self.long, self.short]

from datetime import datetime
from typing import List, Union

import pandas as pd


class FundingRate:
    def __init__(self):
        self.timestamp: Union[int, None] = None
        self.price: Union[float, None] = None
        self.rate: Union[float, None] = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp) if self.timestamp is not None else None

    @staticmethod
    def to_dataframe(funding_rates: List) -> pd.DataFrame:
        data = [FundingRate.to_list(funding_rate) for funding_rate in funding_rates]
        columns = ['timestamp', 'price', 'rate']
        df = pd.DataFrame(data=data, columns=columns)
        df["timestamp"] = df.timestamp.astype(int)
        df = df.set_index('timestamp')

        return df

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def from_list(data: List):
        instance = FundingRate()

        instance.timestamp = int(data[0])
        instance.price = float(data[1])
        instance.rate = float(data[2])

        return instance

    def to_list(self):
        return [self.timestamp, self.price, self.rate]

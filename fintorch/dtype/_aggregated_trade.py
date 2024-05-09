from datetime import datetime
from typing import Union, List

import pandas as pd


class AggregatedTrade:
    def __init__(self):
        self.timestamp: Union[float, None] = None
        self.price: Union[float, None] = None
        self.quantity: Union[float, None] = None
        self.side: Union[int, None] = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp) if self.timestamp is not None else None

    @staticmethod
    def to_dataframe(aggregated_trades: List) -> pd.DataFrame:
        data = [agg_trades.to_list() for agg_trades in aggregated_trades]
        columns = ['timestamp', 'price', 'quantity', "side"]
        df = pd.DataFrame(data=data, columns=columns)
        df = df.set_index('timestamp')

        return df

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def from_list(data: List):
        instance = AggregatedTrade()

        instance.timestamp = int(data[0])
        instance.price = float(data[1])
        instance.quantity = float(data[2])
        instance.side = int(data[3])

        return instance

    def to_list(self):
        return [self.timestamp, self.price, self.quantity, self.side]

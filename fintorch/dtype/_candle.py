from datetime import datetime
from typing import List, Union

import pandas as pd


class Candle:
    def __init__(self):
        self.timestamp: Union[int, None] = None
        self.open: Union[float, None] = None
        self.high: Union[float, None] = None
        self.low: Union[float, None] = None
        self.close: Union[float, None] = None
        self.volume: Union[float, None] = None
        self.trade: Union[int, None] = None

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp) if self.timestamp is not None else None

    @staticmethod
    def from_csv(path: str) -> List:
        candles = []
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                tokens = line.replace("\n", "").split(',')
                candle = Candle.from_list(tokens)
                candles.append(candle)

        return candles

    @staticmethod
    def to_csv(candles: List, path: str):
        df = Candle.to_dataframe(candles)
        df.to_csv(path)

    @staticmethod
    def to_dataframe(candles: List) -> pd.DataFrame:
        data = [candle.to_list() for candle in candles]
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade']
        df = pd.DataFrame(data=data, columns=columns)
        df = df.set_index('timestamp')

        return df

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def from_list(data: List):
        instance = Candle()

        instance.timestamp = int(data[0])
        instance.open = float(data[1])
        instance.high = float(data[2])
        instance.low = float(data[3])
        instance.close = float(data[4])
        instance.volume = float(data[5])
        instance.trade = int(data[6])

        return instance

    def to_list(self):
        return [self.timestamp, self.open, self.high, self.low, self.close, self.volume, self.trade]

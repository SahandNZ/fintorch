from datetime import datetime
from typing import List

import pandas as pd


class Candle:
    def __init__(self):
        self.timestamp: int = None
        self.datetime: datetime = None
        self.open: float = None
        self.high: float = None
        self.low: float = None
        self.close: float = None
        self.volume: float = None
        self.trade: int = None

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
    def to_csv(candles: List, path: str, mode: str):
        with open(path, mode) as file:
            if 'w' in mode:
                file.write(Candle.csv_header())
            for candle in candles:
                line = ','.join(str(item) for item in candle.to_list()) + '\n'
                file.write(line)

    @staticmethod
    def to_dataframe(candles: List) -> pd.DataFrame:
        data = [candle.to_list() for candle in candles]
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade']
        df = pd.DataFrame(data=data, columns=columns)
        df.insert(0, 'datetime', [datetime.fromtimestamp(timestamp) for timestamp in df.timestamp])
        df = df.set_index('timestamp')

        return df

    @staticmethod
    def csv_header():
        return 'timestamp,open,high,low,close,volume,trade\n'

    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def from_list(data: List):
        instance = Candle()

        instance.timestamp = int(data[0])
        instance.datetime = datetime.fromtimestamp(instance.timestamp)
        instance.open = float(data[1])
        instance.high = float(data[2])
        instance.low = float(data[3])
        instance.close = float(data[4])
        instance.volume = float(data[5])
        instance.trade = int(data[6])

        return instance

    def to_list(self):
        return [self.timestamp, self.open, self.high, self.low, self.close, self.volume, self.trade]

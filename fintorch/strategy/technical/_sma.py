from datetime import datetime

import pandas as pd

from .._strategy import Strategy
from ...dtype import Order
from ...enum import OrderSide, TimeFrame
from ...exchange import Exchange


class SmaStrategy(Strategy):
    def __init__(self, symbol: str, time_frame: TimeFrame, length: int = 20):
        super().__init__(name="SMA Strategy", short_name="SMA", symbol=symbol, time_frame=time_frame)
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def next(self, exchange: Exchange) -> None:
        df = exchange.future.data.get_candles_dataframe(symbol=self.symbol, time_frame=self.time_frame)
        df = self.process_df(df=df)

        if 0 < len(df):
            position = exchange.future.trade.get_position(symbol=self.symbol)
            if df.cross.iloc[-1]:
                order = Order()
                order.symbol = self.symbol
                order.side = OrderSide.BUY if df.above.iloc[-1] else OrderSide.SELL
                order.percentage = 200 if position.is_open else 100

                exchange.future.trade.set_order(order=order)

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df.insert(0, "datetime", [datetime.fromtimestamp(ts) for ts in df.index])
        df['sma'] = df.close.rolling(self.length).mean()
        df['above'] = df.sma < df.close
        df['cross'] = df.above.diff()

        return df

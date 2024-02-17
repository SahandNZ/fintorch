import pandas as pd

from ...dtype import Order
from ...enum import OrderSide, TimeFrame

from ...exchange import Exchange
from ..strategy import Strategy


class SmaStrategy(Strategy):
    def __init__(self, symbol: str, time_frame: TimeFrame, length: int = 20):
        super().__init__(name="SMA Strategy", short_name="SMA", symbol=symbol, time_frame=time_frame)
        self.__length: int = length

    @property
    def length(self) -> int:
        return self.__length

    def prepare(self, exchange: Exchange) -> None:
        exchange.future.trade.set_leverage(symbol=self.symbol, leverage=10)

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sma'] = df.close.rolling(self.length).mean()
        df['up'] = df.sma < df.close
        df['cross'] = df.up.diff()
        df = df.dropna()

        return df

    def next(self, exchange: Exchange) -> None:
        df = exchange.future.data.get_candles_dataframe(symbol=self.symbol, time_frame=self.time_frame)
        df = self.process_df(df=df)

        if df.cross.iloc[-1]:
            order = Order()
            order.symbol = self.symbol
            order.side = OrderSide.BUY if df.up.iloc[-1] else OrderSide.SELL
            order.quantity = 0.001

            exchange.future.trade.set_order(order=order)

    def __str__(self):
        return f"{self.short_name} {self.symbol} {str(self.time_frame)}"

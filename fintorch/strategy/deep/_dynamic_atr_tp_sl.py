from abc import ABC

from . import DeepStrategy
from ...deep.module import Module
from ...dtype import Position


class DynamicAtrTpSlDeepStrategy(DeepStrategy, ABC):
    def __init__(self, module: Module, window: int = 5, take_profit_factor: float = 2, stop_loss_factor: float = 1):
        super().__init__(name="Dynamic-Atr TP-SL Deep Strategy", short_name="D-ATR TP-SL", module=module)
        self.__window: int = window
        self.__take_profit_factor: float = take_profit_factor
        self.__stop_loss_factor: float = stop_loss_factor

    @property
    def window(self) -> int:
        return self.__window

    @property
    def take_profit_factor(self) -> float:
        return self.__take_profit_factor

    @property
    def stop_loss_factor(self) -> float:
        return self.__stop_loss_factor

    def on_new_candle(self) -> None:
        df = self.process_df()

        if 0 < len(df):
            side = df.side.iloc[-1]
            position = self.future.trade.get_position(symbol=self.symbol)

            if position.side != side:
                if position.is_open:
                    self.future.trade.set_exit_order(symbol=self.symbol, percentage=100, comment="Exit")

                self.future.trade.set_entry_order(symbol=self.symbol, side=side, percentage=100, comment="Entry")

    def on_opened_position(self, position: Position) -> None:
        dc = self.future.data.get_data_collection(symbols=[self.symbol], time_frames=[self.time_frame])
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.time_frame).copy()
        df["atr"] = df.high / df.low - 1
        df["mean-atr"] = df.atr.rolling(window=self.window).mean()

        take_profit_rate = df["mean-atr"].iloc[-1] * self.take_profit_factor
        stop_loss_rate = df["mean-atr"].iloc[-1] * self.stop_loss_factor

        take_profit_price = position.entry_price * (1 + int(position.side) * take_profit_rate)
        stop_loss_price = position.entry_price * (1 - int(position.side) * stop_loss_rate)

        self.future.trade.set_exit_order(symbol=self.symbol, percentage=100, price=take_profit_price, comment="TP")
        self.future.trade.set_exit_order(symbol=self.symbol, percentage=100, stop_price=stop_loss_price, comment="SL")

    def on_closed_position(self, position: Position) -> None:
        self.future.trade.cancel_all_orders(symbol=position.symbol)

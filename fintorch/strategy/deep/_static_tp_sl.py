from abc import ABC

from . import SimpleDeepStrategy, DeepStrategy
from ...deep.module import Module
from ...dtype import Position


class StaticTpSlDeepStrategy(DeepStrategy, ABC):
    def __init__(self, module: Module, take_profit_rate: float = 0.1, stop_loss_rate: float = 0.05):
        super().__init__(name="Static TP-SL Deep Strategy", short_name="S TP-SL", module=module)
        self.__take_profit_rate: float = take_profit_rate
        self.__stop_loss_rate: float = stop_loss_rate

    @property
    def take_profit_rate(self) -> float:
        return self.__take_profit_rate

    @property
    def stop_loss_rate(self) -> float:
        return self.__stop_loss_rate

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
        take_profit_price = position.entry_price * (1 + int(position.side) * self.take_profit_rate)
        stop_loss_price = position.entry_price * (1 - int(position.side) * self.stop_loss_rate)

        self.future.trade.set_exit_order(symbol=self.symbol, percentage=100, price=take_profit_price, comment="TP")
        self.future.trade.set_exit_order(symbol=self.symbol, percentage=100, stop_price=stop_loss_price, comment="SL")

    def on_closed_position(self, position: Position) -> None:
        self.future.trade.cancel_all_orders(symbol=position.symbol)

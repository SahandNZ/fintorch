from fintorch.deep.module import Module
from fintorch.dtype import Position
from ._simple import SimpleStrategy


class StaticTpSlStrategy(SimpleStrategy):
    def __init__(self, module: Module, take_profit_rate: float = 0.1, stop_loss_rate: float = 0.05):
        super().__init__(name="Static TP-SL Strategy", short_name="S TP-SL", module=module)

        self.__take_profit_rate: float = take_profit_rate
        self.__stop_loss_rate: float = stop_loss_rate

    @property
    def take_profit_rate(self) -> float:
        return self.__take_profit_rate

    @property
    def stop_loss_rate(self) -> float:
        return self.__stop_loss_rate

    def on_opened_position(self, position: Position) -> None:
        take_profit_price = position.entry_price * (1 + int(position.side) * self.take_profit_rate)
        stop_loss_price = position.entry_price * (1 - int(position.side) * self.stop_loss_rate)

        self.future.trade.set_exit_order(position=position, percentage=100, price=take_profit_price, comment="TP")
        self.future.trade.set_exit_order(position=position, percentage=100, stop_price=stop_loss_price, comment="SL")

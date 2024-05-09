from abc import ABC

from ._deep import DeepStrategy
from ...deep.module import Module
from ...dtype import Position


class SimpleDeepStrategy(DeepStrategy, ABC):
    def __init__(self, module: Module):
        super().__init__(name="Simple Deep Strategy", short_name="Simple", module=module)

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
        pass

    def on_closed_position(self, position: Position) -> None:
        pass

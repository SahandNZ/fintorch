import numpy as np

from ._strategy import Strategy
from ..deep.module import Module
from ..dtype import Order


class PassiveLimitStrategy(Strategy):
    def __init__(self, module: Module) -> None:
        super().__init__(
            name="Passive Limit Strategy",
            short_name="PL",
            module=module,
            indicators=[],
            height_ratios=[1]
        )
        
        self.entry_order: Order = None

    def on_new_candle(self) -> None:
        if 0 < len(self.df):
            # market data
            side = self.df.side.iloc[-1]
            if np.isnan(side):
                return
            
            bmax, bmin = self.df.bmax.iloc[-1], self.df.bmin.iloc[-1]
            limit = bmin if 1 == side else bmax

            # trade data
            position = self.future.trade.get_position(symbol=self.symbol)
            open_orders = self.future.trade.get_open_orders(symbol=self.symbol)
            if self.entry_order not in open_orders:
                self.entry_order = None

            # update entry and exit orders 
            if position.side != side:
                # there is an open position
                if position.is_open:
                    if 0 < position.profit_percentage:
                        self.future.trade.set_exit_order(
                            position=position,
                            percentage=100,
                            comment="Market Exit"
                        )

                # there is no open position
                if not position.is_open:
                    if self.entry_order is None:
                        self.entry_order = self.future.trade.set_entry_order(
                            symbol=self.symbol,
                            side=side,
                            percentage=100,
                            price=limit,
                            comment="Limit Entry"
                        )
                        
                    if self.entry_order is not None and (self.entry_order.side != side or self.entry_order.price != limit):
                        self.future.trade.cancel_order(symbol=self.symbol, order_id=self.entry_order.id)
                        self.entry_order = self.future.trade.set_entry_order(
                            symbol=self.symbol,
                            side=side,
                            percentage=100,
                            price=limit,
                            comment="Limit Entry"
                        )
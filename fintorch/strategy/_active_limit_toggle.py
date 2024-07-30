import numpy as np

from ._strategy import Strategy
from ..deep.module import Module
from ..dtype import Order, Position


class ActiveLimitToggleStrategy(Strategy):
    def __init__(self, module: Module) -> None:
        super().__init__(
            name="Active Limit Toggle Strategy",
            short_name="AL-TOG",
            module=module,
            indicators=["trend", "bmax", "bmin"],
            height_ratios=[1]
        )
        
        self.entry_order: Order = None
        self.take_profit_order: Order = None
        self.stop_loss_order: Order = None
        self.is_risk_free_done: bool = False
        self.is_head_to_head_done: bool = False

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
            if self.take_profit_order not in open_orders:
                self.take_profit_order = None
            if self.stop_loss_order not in open_orders:
                self.stop_loss_order = None
                
                
            # update take profit order (Head to Head order)
            if position.is_open and position.profit_percentage <= -3.3:
                head_to_head_price = position.entry_price
                if not self.is_head_to_head_done and self.take_profit_order.price != head_to_head_price:
                    self.is_head_to_head_done = True
                    self.future.trade.cancel_order(symbol=self.symbol, order_id=self.take_profit_order.id)
                    self.future.trade.set_exit_order(
                        position=position,
                        percentage=100,
                        price=head_to_head_price,
                        comment="Take Profit"
                    )

            # update stop loss order (Risk Free order)
            if position.is_open and 5 <= position.profit_percentage:
                risk_free_price = position.entry_price * (1 + int(position.side) * 0.025)
                if not self.is_risk_free_done and self.stop_loss_order.stop_price != risk_free_price:
                    self.is_risk_free_done = True
                    self.future.trade.cancel_order(symbol=self.symbol, order_id=self.stop_loss_order.id)
                    self.stop_loss_order = self.future.trade.set_exit_order(
                        position=position,
                        percentage=100,
                        stop_price=risk_free_price,
                        comment="Stop Loss"
                    )

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
                        

    def on_opened_position(self, position: Position) -> None:
        self.is_risk_free_done = False
        self.is_head_to_head_done = False 
        
        take_profit_price = position.entry_price * (1 + int(position.side) * 0.1)
        stop_loss_price = position.entry_price * (1 - int(position.side) * 0.1)

        self.take_profit_order = self.future.trade.set_exit_order(
            position=position,
            percentage=100,
            price=take_profit_price,
            comment="Take Profit"
        )
        
        self.stop_loss_order = self.future.trade.set_exit_order(
            position=position,
            percentage=100,
            stop_price=stop_loss_price,
            comment="Stop Loss"
        )

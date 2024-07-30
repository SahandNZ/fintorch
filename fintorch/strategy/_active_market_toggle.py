from typing import Dict

import numpy as np

from ._error import NanInSideColumnException
from ._strategy import Strategy
from ..deep.module import Module
from ..dtype import Order, Position


class ActiveMarketToggleStrategy(Strategy):
    def __init__(self, module: Module) -> None:
        super().__init__(
            name="Active Market Toggle Strategy",
            short_name="TOG",
            module=module,
            indicators=[],
            height_ratios=[1]
        )
        
        # constants
        self.take_profit_rate: float = 0.075
        self.stop_loss_rate: float = 0.05
        self.head_to_head_threshold_rate: float = self.take_profit_rate / 3
        self.risk_free_threshold_rate: float = self.stop_loss_rate / 2
        
        # state
        self.take_profit_order: Order = None
        self.stop_loss_order: Order = None
        self.is_risk_free_done: bool = False
        self.is_head_to_head_done: bool = False
        
    def get_state_dict(self) -> Dict:
        state_dict = super().get_state_dict()
        state_dict.update({
            "take_profit_order": self.take_profit_order,
            "stop_loss_order": self.stop_loss_order,
            "is_risk_free_done": self.is_risk_free_done,
            "is_head_to_head_done": self.is_head_to_head_done
        })
        
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        super().load_state_dict(state_dict=state_dict)
        self.take_profit_order = state_dict.get("take_profit_order", None)
        self.stop_loss_order = state_dict.get("stop_loss_order", None)
        self.is_risk_free_done = state_dict.get("is_risk_free_done", False)
        self.is_head_to_head_done = state_dict.get("is_head_to_head_done", False)

    def clear_state(self) -> None:
        super().clear_state()
        self.take_profit_order: Order = None
        self.stop_loss_order: Order = None
        self.is_risk_free_done: bool = False
        self.is_head_to_head_done: bool = False

    def on_new_candle(self) -> None:
        if 0 < len(self.df):
            side = self.df.side.iloc[-1]
            
            if np.isnan(side).any():
                raise NanInSideColumnException("NaN or Inf values found in side column of dataframe.")

            # trade data
            position = self.future.trade.get_position(symbol=self.symbol)
            open_orders = self.future.trade.get_open_orders(symbol=self.symbol)
            open_orders_id = set([order.id for order in open_orders])
            if self.take_profit_order is not None and self.take_profit_order.id not in open_orders_id:
                self.take_profit_order = None
            if self.stop_loss_order is not None and self.stop_loss_order.id not in open_orders_id:
                self.stop_loss_order = None

            # update take profit order (Head to Head order)
            if position.is_open and position.profit_rate <= -self.head_to_head_threshold_rate:
                head_to_head_price = position.entry_price
                if not self.is_head_to_head_done and self.take_profit_order.price != head_to_head_price:
                    self.is_head_to_head_done = True
                    self.future.trade.cancel_order(symbol=self.symbol, order_id=self.take_profit_order.id)
                    self.take_profit_order = self.future.trade.set_exit_order(
                        position=position,
                        percentage=100,
                        price=head_to_head_price,
                        comment="Take Profit"
                    )

            # update stop loss order (Risk Free order)
            if position.is_open and self.risk_free_threshold_rate <= position.profit_rate:
                risk_free_price = position.entry_price
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
                    if self.stop_loss_rate / 3 <= position.profit_percentage <= self.take_profit_rate / 3:
                        self.future.trade.set_exit_order(
                            position=position,
                            percentage=100,
                            comment="Exit"
                        )

                # there is no open position
                if not position.is_open:
                    self.entry_order = self.future.trade.set_entry_order(
                        symbol=self.symbol,
                        side=side,
                        percentage=100,
                        comment="Entry"
                    )

    def on_opened_position(self, position: Position) -> None:
        self.is_risk_free_done = False
        self.is_head_to_head_done = False

        take_profit_price = position.entry_price * (1 + int(position.side) * self.take_profit_rate)
        stop_loss_price = position.entry_price * (1 - int(position.side) * self.stop_loss_rate)

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

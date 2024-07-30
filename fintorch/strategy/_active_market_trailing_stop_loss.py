from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from ._error import NanInSideColumnException
from ._strategy import Strategy
from ..deep.module import Module
from ..dtype import Order, Position, DataCollection
from ..enum import PositionSide


class ActiveMarketTrailingStopLossStrategy(Strategy):
    def __init__(self, module: Module) -> None:
        super().__init__(
            name="Active Market Trailing Stop Loss Strategy",
            short_name="TSL",
            module=module,
            indicators=[],
            height_ratios=[1]
        )

        self.offset: float = 0.0075
        self.activation_profit_rate: float = 0.02
        self.init_stop_loss_rate: float = 0.05
        self.stop_loss_order: Order = None

    def get_state_dict(self) -> Dict:
        state_dict = super().get_state_dict()
        state_dict.update({"stop_loss_order": self.stop_loss_order})

        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        super().load_state_dict(state_dict=state_dict)
        self.stop_loss_order = state_dict.get("stop_loss_order", None)

    def clear_state(self) -> None:
        super().clear_state()
        self.stop_loss_order: Order = None

    def process_dc_to_interval_df(self, dc: DataCollection) -> pd.DataFrame:
        df = dc.get_candles_df(symbol=self.symbol, time_frame=self.interval).copy()
        df.insert(0, "datetime", [datetime.fromtimestamp(ts) for ts in df.index])
        df["long"] = df.high * (1 - self.offset)
        df["short"] = df.low * (1 + self.offset)
        df = df.dropna()

        return df

    def on_new_interval(self) -> None:
        position = self.future.trade.get_position(symbol=self.symbol)
        if position.is_open and self.activation_profit_rate < position.profit_rate:
            df = self.interval_df
            side = position.side
            current_stop_loss_price = self.stop_loss_order.stop_price
            new_stop_loss_price = df.long.iloc[-1] if PositionSide.LONG == position.side else df.short.iloc[-1]

            if current_stop_loss_price * side < new_stop_loss_price * side:
                # print("Previous stop order was canceled and new stop order submitted!")
                self.future.trade.cancel_order(symbol=self.symbol, order_id=self.stop_loss_order.id)
                self.stop_loss_order = self.future.trade.set_exit_order(
                    position=position,
                    percentage=100,
                    stop_price=new_stop_loss_price,
                    comment="TR-SL"
                )

    def on_new_candle(self) -> None:
        if 0 < len(self.df):
            side = self.df.side.iloc[-1]
            position = self.future.trade.get_position(symbol=self.symbol)
            
            if np.isnan(side).any():
                raise NanInSideColumnException("NaN or Inf values found in side column of dataframe.")

            if not position.is_open and position.side != side:
                self.future.trade.set_entry_order(
                    symbol=self.symbol,
                    side=side,
                    percentage=100,
                    comment="Entry"
                )

    def on_opened_position(self, position: Position) -> None:
        stop_loss_price = position.entry_price * (1 - int(position.side) * self.init_stop_loss_rate)
        self.stop_loss_order = self.future.trade.set_exit_order(
            position=position,
            percentage=100,
            stop_price=stop_loss_price,
            comment="TR-SL"
        )

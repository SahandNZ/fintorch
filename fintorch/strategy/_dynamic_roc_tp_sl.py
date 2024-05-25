import numpy as np
import pandas as pd

from ._simple import SimpleStrategy
from ..deep.module import Module
from ..dtype import Position, DataCollection


class DynamicRocTpSlStrategy(SimpleStrategy):
    def __init__(self, module: Module, window: int = 5, take_profit_factor: float = 5, stop_loss_factor: float = 2.5):
        super().__init__(name="Dynamic-Roc TP-SL Strategy", short_name="D-ROC TP-SL", module=module)

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

    def process_dc_to_df(self, dc: DataCollection) -> pd.DataFrame:
        df = super().process_dc_to_df(dc=dc)
        df["abs-roc"] = np.abs(df.close / df.open - 1)
        df["mean-abs-roc"] = df["abs-roc"].rolling(window=self.window).mean()

        return df

    def on_opened_position(self, position: Position) -> None:
        volatility = self.df["mean-abs-roc"].iloc[-1]
        take_profit_rate = volatility * self.take_profit_factor
        stop_loss_rate = volatility * self.stop_loss_factor

        take_profit_price = position.entry_price * (1 + int(position.side) * take_profit_rate)
        stop_loss_price = position.entry_price * (1 - int(position.side) * stop_loss_rate)

        self.future.trade.set_exit_order(position=position, percentage=100, price=take_profit_price, comment="TP")
        self.future.trade.set_exit_order(position=position, percentage=100, stop_price=stop_loss_price, comment="SL")

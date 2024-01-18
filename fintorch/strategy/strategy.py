from abc import abstractmethod
from typing import List

import pandas as pd
from fintorch.cross_validation.fold import Fold
from fintorch.position import Position

from fintorch.component import Component


class Strategy(Component):
    def __init__(self, name: str, short_name: str, show_progress_bar: bool):
        super().__init__(name=name, short_name=short_name, description="")
        self.__show_progress_bar: bool = show_progress_bar

    @property
    def show_progress_bar(self) -> bool:
        return self.__show_progress_bar

    @abstractmethod
    def prepare_dataframe(self, fold: Fold) -> pd.DataFrame:
        raise NotImplemented()

    @abstractmethod
    def backtest(self, fold: Fold) -> List[Position]:
        raise NotImplemented()

    @staticmethod
    def _create_position(df: pd.DataFrame, entry_index: int, exit_index: int, exit_price: float) -> Position:
        bars = exit_index - entry_index
        entry_timestamp = df.timestamp.iloc[entry_index]
        exit_timestamp = df.timestamp.iloc[exit_index]
        side = df.side.iloc[entry_index]
        entry_price = df.open.iloc[entry_index]
        entry_percentage = 100
        maximum_met_price = df.high.iloc[entry_index:exit_index].max()
        minimum_met_price = df.low.iloc[entry_index:exit_index].min()
        position = Position(bars=bars, entry_timestamp=entry_timestamp, exit_timestamp=exit_timestamp,
                            side=side, entry_percentage=entry_percentage, entry_price=entry_price,
                            exit_price=exit_price, maximum_met_price=maximum_met_price,
                            minimum_met_price=minimum_met_price)

        return position

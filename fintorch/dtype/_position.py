from datetime import datetime
from typing import Union

from ..enum import PositionSide, PositionType


class Position:
    def __init__(self):
        self.id: Union[str, None] = None
        self.symbol: Union[str, None] = None
        self.leverage: Union[int, None] = None
        self.type: Union[PositionType, None] = None

        # opening properties
        self.side: Union[PositionSide, None] = None
        self.entry_price: Union[float, None] = None
        self.entry_timestamp: Union[int, None] = None
        self.entry_percentage: Union[int, None] = None

        # while open properties
        self.current_timestamp: Union[int, None] = None
        self.current_price: Union[float, None] = None
        self.lowest_met_price: Union[float, None] = None
        self.highest_met_price: Union[float, None] = None

        # closing properties
        self.exit_price: Union[float, None] = None
        self.exit_timestamp: Union[int, None] = None
        self.exit_percentage: Union[float, None] = None

        # last phase properties
        self.quantity: Union[float, None] = None

    @property
    def entry_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.entry_timestamp) if self.entry_timestamp is not None else None

    @property
    def is_opened(self) -> bool:
        return 0 < self.entry_percentage

    @property
    def is_closed(self) -> bool:
        return 100 == self.exit_percentage

    @property
    def is_open(self) -> bool:
        return self.is_opened and not self.is_closed

    @property
    def exit_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.exit_timestamp) if self.exit_timestamp is not None else None

    @property
    def duration(self) -> float:
        exit_timestamp = self.exit_timestamp or self.current_timestamp
        return exit_timestamp - self.entry_timestamp

    @property
    def profit_percentage(self) -> float:
        exit_price = self.exit_price or self.current_price
        return self._calculate_pnl_percentage(exit_price=exit_price)

    @property
    def run_up_percentage(self) -> float:
        best_met_price = self.highest_met_price if PositionSide.LONG == self.side else self.lowest_met_price
        return self._calculate_pnl_percentage(exit_price=best_met_price)

    @property
    def draw_down_percentage(self) -> float:
        worst_met_price = self.lowest_met_price if PositionSide.LONG == self.side else self.highest_met_price
        return abs(self._calculate_pnl_percentage(exit_price=worst_met_price))
    
    @property
    def profit_rate(self) -> float:
        return round(self.profit_percentage / 100, 6)

    @property
    def margin(self) -> Union[float, None]:
        if self.quantity is not None:
            return round(self.quantity * self.entry_price / self.leverage, 2)

    @property
    def profit(self) -> Union[float, None]:
        return self._calculate_pnl(percentage=self.profit_percentage)

    @property
    def run_up(self) -> Union[float, None]:
        return self._calculate_pnl(percentage=self.run_up_percentage)

    @property
    def draw_down(self) -> Union[float, None]:
        return self._calculate_pnl(percentage=self.draw_down_percentage)

    def paid_fee(self, fee_percentage: float) -> Union[float, None]:
        exit_price = self.exit_price or self.current_price
        entry_fee = self.entry_price * self.quantity * fee_percentage / 100
        exit_fee = exit_price * self.quantity * fee_percentage / 100
        return round(entry_fee + exit_fee, 2)

    def _calculate_pnl_percentage(self, exit_price: float) -> float:
        pnl_percentage = (exit_price / self.entry_price - 1) * int(self.side) * self.leverage * 100
        return round(pnl_percentage, 2)

    def _calculate_pnl(self, percentage: float) -> Union[float, None]:
        if self.margin is not None:
            return round(self.margin * percentage / 100, 2)

    def __str__(self):
        return (("Position {} {}\n"
                 "\t- {:<32}{}\n"
                 "\t- {:<32}{}\n"
                 "\t- {:<32}{}\n"
                 "\t- {:<32}{}\n"
                 "\t- {:<32}{}\n"
                 "\t- {:<32}{}\n"
                 "\t- {:<32}{}\n"
                 .format(self.symbol, str(self.side),
                         "Leverage", self.leverage,
                         "Quantity", self.quantity,
                         "Entry Price", self.entry_price,
                         "Entry Date", self.entry_datetime,
                         "Exit Price", self.exit_price,
                         "Exit Date", self.exit_datetime,
                         "Profit Percentage", self.profit_percentage)))

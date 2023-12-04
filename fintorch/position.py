from datetime import datetime
from typing import List

from tabulate import tabulate


class Position:
    def __init__(self, bars: int = None, entry_timestamp: int = None, exit_timestamp: int = None,
                 side: int = None, entry_percentage: int = None, entry_price: float = None, exit_price: float = None,
                 minimum_met_price: float = None, maximum_met_price: float = None):
        # private properties
        self.__bars: int = bars
        self.__entry_timestamp: int = entry_timestamp
        self.__exit_timestamp: int = exit_timestamp
        self.__side: int = side
        self.__entry_percentage: int = entry_percentage
        self.__entry_price: float = entry_price
        self.__exit_price: float = exit_price
        self.__minimum_met_price: float = minimum_met_price
        self.__maximum_met_price: float = maximum_met_price

        # quantity related properties
        self.equity: float = None
        self.quantity: float = None
        self.profit: float = None
        self.run_up: float = None
        self.drawdown: float = None
        self.buy_and_hold: float = None

    @property
    def bars(self) -> int:
        return self.__bars

    @property
    def entry_timestamp(self) -> int:
        return self.__entry_timestamp

    @property
    def entry_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.entry_timestamp) if self.entry_timestamp is not None else None

    @property
    def exit_timestamp(self) -> float:
        return self.__exit_timestamp

    @property
    def exit_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.exit_timestamp) if self.exit_timestamp is not None else None

    @property
    def side(self) -> int:
        return self.__side

    @property
    def entry_price(self) -> float:
        return self.__entry_price

    @property
    def entry_percentage(self) -> float:
        return self.__entry_percentage

    @property
    def exit_price(self) -> float:
        return self.__exit_price

    @property
    def minimum_met_price(self) -> float:
        return self.__minimum_met_price

    @property
    def maximum_met_price(self) -> float:
        return self.__maximum_met_price

    @property
    def time_frame(self) -> int:
        return (self.entry_timestamp - self.exit_timestamp) // self.bars

    @property
    def best_met_price(self) -> float:
        return self.maximum_met_price if 1 == self.side else self.minimum_met_price

    @property
    def worst_met_price(self) -> float:
        return self.minimum_met_price if 1 == self.side else self.maximum_met_price

    @property
    def profit_percentage(self) -> float:
        return round((self.exit_price / self.entry_price - 1) * self.side * 100, 4)

    @property
    def run_up_percentage(self) -> float:
        return round((self.best_met_price / self.entry_price - 1) * self.side * 100, 4)

    @property
    def drawdown_percentage(self) -> float:
        return round((self.worst_met_price / self.entry_price - 1) * self.side * 100, 4)

    @property
    def buy_and_hold_percentage(self) -> float:
        return round(self.profit_percentage * self.side, 2)

    def set_quantity(self, quantity: float):
        self.equity = round(self.entry_price * quantity, 2)
        self.set_equity(self.equity)

    def set_equity(self, equity: float):
        self.equity = equity
        self.quantity = self.equity / self.entry_price
        self.profit = round(self.profit_percentage / 100 * self.equity, 2)
        self.run_up = round(self.run_up_percentage / 100 * self.equity, 2)
        self.drawdown = round(self.drawdown_percentage / 100 * self.equity, 2)
        self.buy_and_hold = round(self.buy_and_hold_percentage / 100 * self.equity, 2)

    def to_list(self):
        return [self.entry_datetime, self.exit_datetime, self.side, self.entry_price,
                self.exit_price, self.entry_percentage, self.profit_percentage, self.run_up_percentage,
                self.drawdown_percentage, self.buy_and_hold_percentage]

    @staticmethod
    def tabulate(positions: List):
        values = [position.to_list() for position in positions]
        headers = ["Entry Datetime", "Exit Datetime", "Side", "Entry Price", "Exit Price",
                   "Entry %", "Profit %", "Run-up %", "Drawdown %", "Buy & Hold %"]
        table = tabulate(values, headers=headers, tablefmt="pretty")
        print(table)

    def __str__(self) -> str:
        return "Position:" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}" \
               "\n\t- {:<20}{}\n" \
            .format("Bars", self.bars,
                    "Entry Datetime", self.entry_datetime,
                    "Exit Datetime", self.exit_datetime,
                    "Side", self.side,
                    "Entry price", self.entry_price,
                    "Entry %", self.entry_percentage,
                    "Exit price", self.exit_price,
                    "Profit %", self.profit_percentage,
                    "Draw down %", self.drawdown_percentage,
                    "Run up %", self.run_up_percentage)
